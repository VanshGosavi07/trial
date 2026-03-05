"""
Agent 2: MCP Data Agent
────────────────────────
Wraps ZohoCRMMCPClient.
Given a QueryPlan (from QueryAgent), executes every SQL component
and returns a DataBundle — a list of QueryResult dicts.

QueryResult schema:
{
  "id":          str    — component id,
  "label":       str    — human-readable label,
  "output_type": str    — kpi | bar | line | pie | funnel | …,
  "description": str    — component description,
  "columns":     list   — ["col1", "col2", …],
  "rows":        list   — [{"col1": val, …}, …],
  "error":       str | None
}
"""

import csv
import io
import json
import re
import sys
from pathlib import Path

# Retry on transport / timeout errors only — NOT on permanent SQL errors
_MAX_RETRIES = 2

# Error summary words that indicate a permanent SQL failure (no point retrying)
_PERMANENT_ERROR_KEYWORDS = frozenset({
    "INVALID_COLUMN", "INVALID_TABLE", "TABLE_NOT_FOUND",
    "SYNTAX_ERROR", "PERMISSION_DENIED", "COLUMN_NOT_FOUND",
})

# ── Import existing MCP client ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from mcp_chat_client import ZohoCRMMCPClient


class MCPDataAgent:
    """
    Data Fetching Agent — executes SQL via Zoho Analytics MCP.
    """

    def __init__(self):
        self._mcp: ZohoCRMMCPClient | None = None
        # Cache of {table_name: [col1, col2, …]} — populated on first schema fetch
        self._schema_cache: dict[str, list[str]] = {}

    # ─────────────────────────────────────────────────────────────────────────
    def ensure_connected(self) -> bool:
        if self._mcp and self._mcp.initialized:
            return True
        print("[MCPDataAgent] Connecting to Zoho Analytics MCP…")
        self._mcp = ZohoCRMMCPClient()
        ok = self._mcp.initialize()
        if ok:
            print(f"[MCPDataAgent] Connected — workspace: {self._mcp.workspace_name}")
        else:
            print("[MCPDataAgent] Connection FAILED")
        return ok

    @property
    def available_tables(self) -> list[str]:
        """List of table names discovered from the workspace."""
        if not self._mcp or not self._mcp.initialized:
            return []
        return [v["viewName"]
                for v in self._mcp.all_views
                if v.get("viewType") == "Table"]

    @property
    def workspace_name(self) -> str:
        return self._mcp.workspace_name if self._mcp else "Unknown"

    # ─────────────────────────────────────────────────────────────────────────
    # Schema Discovery
    # ─────────────────────────────────────────────────────────────────────────

    def get_table_schemas(self, table_names: list[str]) -> dict[str, list[str]]:
        """
        Return {table_name: [col1, col2, …]} for every table in *table_names*.

        Fetches each table with ``SELECT * FROM "T" LIMIT 1`` the first time and
        caches the result for the session lifetime — no repeated API calls.
        """
        schemas: dict[str, list[str]] = {}
        for tname in table_names:
            if tname in self._schema_cache:
                schemas[tname] = self._schema_cache[tname]
            else:
                cols = self._fetch_columns(tname)
                if cols:
                    self._schema_cache[tname] = cols
                    schemas[tname] = cols
        return schemas

    def _fetch_columns(self, table_name: str) -> list[str]:
        """
        Discover column names for one table by running ``SELECT * LIMIT 1``.
        Returns [] on any failure — schema will be omitted for that table.
        """
        if not self.ensure_connected():
            return []
        try:
            sql = f'SELECT * FROM "{table_name}" LIMIT 1'
            raw = self._mcp.run_sql(sql)
            qr  = self._parse_result(raw)
            cols = qr.get("columns") or []
            if cols:
                print(f"[MCPDataAgent] Schema '{table_name}': {cols}")
            return cols
        except Exception as exc:
            print(f"[MCPDataAgent] Could not fetch schema for '{table_name}': {exc}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    def execute_plan(self, query_plan: dict) -> list[dict]:
        """
        Execute every component SQL in *query_plan*.
        Retries each query up to _MAX_RETRIES times on transport errors.
        Returns a list of QueryResult dicts.
        """
        if not self.ensure_connected():
            return [_empty_result(
                "error", "Connection Error", "kpi", "",
                "Could not connect to Zoho Analytics MCP",
            )]

        components = query_plan.get("components", [])
        n = len(components)
        print(f"[MCPDataAgent] Executing {n} SQL quer{'y' if n == 1 else 'ies'}…")

        results: list[dict] = []
        for comp in components:
            cid   = comp.get("id",          "unknown")
            sql   = (comp.get("sql") or "").strip()
            label = comp.get("label",       cid)
            otype = comp.get("output_type", "bar")
            desc  = comp.get("description", "")

            if not sql or comp.get("_blocked"):
                msg = "SQL blocked by ValidationAgent" if comp.get("_blocked") else "No SQL provided"
                results.append(_empty_result(cid, label, otype, desc, msg))
                continue

            results.append(self._run_with_retry(cid, label, otype, desc, sql))

        return results

    # ─────────────────────────────────────────────────────────────────────────
    def _run_with_retry(
        self,
        cid:   str,
        label: str,
        otype: str,
        desc:  str,
        sql:   str,
    ) -> dict:
        """
        Run SQL with up to _MAX_RETRIES retries.
        Permanent SQL errors (INVALID_COLUMN, SYNTAX_ERROR, …) are not retried
        because re-sending the same SQL will always produce the same error.
        """
        original_sql = sql
        sql = self._sanitize_sql_for_zoho(sql)
        if sql != original_sql:
            print(f"  [{cid}] SQL compatibility fix applied (runtime date predicate removed)")

        preview = f"{sql[:120]}{'…' if len(sql) > 120 else ''}"
        print(f"  [{cid}] SQL: {preview}")

        last_error = "Unknown error"
        for attempt in range(1, _MAX_RETRIES + 2):
            try:
                raw = self._mcp.run_sql(sql)
                qr  = self._parse_result(raw)
                qr.update({"id": cid, "label": label, "output_type": otype, "description": desc})

                if qr.get("error"):
                    last_error = qr["error"]
                    # Never retry permanent SQL errors — they won't change
                    if _is_permanent_error(last_error):
                        print(f"  [{cid}] ⛔ Permanent SQL error: {last_error}")
                        return qr
                    if attempt <= _MAX_RETRIES:
                        print(f"  [{cid}] attempt {attempt} — {last_error} — retrying…")
                        continue
                    print(f"  [{cid}] ⚠️  Failed after {attempt} attempt(s): {last_error}")
                else:
                    print(f"  [{cid}] ✓  {len(qr.get('rows', []))} row(s)")
                return qr

            except Exception as exc:
                last_error = str(exc)
                if attempt <= _MAX_RETRIES:
                    print(f"  [{cid}] attempt {attempt} exception: {exc} — retrying…")
                else:
                    print(f"  [{cid}] ⚠️  Exception after {attempt} attempt(s): {exc}")

        return _empty_result(cid, label, otype, desc, last_error)

    def _sanitize_sql_for_zoho(self, sql: str) -> str:
        """Remove runtime date predicates that are commonly invalid in Zoho SQL."""
        if not sql:
            return sql

        if not re.search(r"CURRENT_DATE|CURDATE\s*\(|NOW\s*\(|GETDATE\s*\(|TODAY\s*\(", sql, re.IGNORECASE):
            return sql

        where_pat = re.compile(
            r"\s+WHERE\s+(.*?)(\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT\s+|\s*$)",
            re.IGNORECASE | re.DOTALL,
        )

        def _replace_where(match: re.Match) -> str:
            cond = match.group(1)
            tail = match.group(2) or ""
            if re.search(r"CURRENT_DATE|CURDATE\s*\(|NOW\s*\(|GETDATE\s*\(|TODAY\s*\(", cond, re.IGNORECASE):
                return tail
            return match.group(0)

        fixed = where_pat.sub(_replace_where, sql)
        return re.sub(r"\s+", " ", fixed).strip()

    # ─────────────────────────────────────────────────────────────────────────
    def execute_sql(self, sql: str, label: str = "Query") -> dict:
        """Execute a single SQL statement and return a QueryResult."""
        if not self.ensure_connected():
            return _empty_result("q", label, "table", label, "MCP not connected")
        try:
            raw = self._mcp.run_sql(sql)
            qr  = self._parse_result(raw)
            qr.update({"id": "q", "label": label, "output_type": "table",
                       "description": label})
            return qr
        except Exception as e:
            return _empty_result("q", label, "table", label, str(e))

    # ─────────────────────────────────────────────────────────────────────────
    def _parse_result(self, raw_result: dict) -> dict:
        """
        Convert ZohoCRMMCPClient.run_sql() output into a normalised
        {"columns": [...], "rows": [...], "error": None} dict.

        Zoho Analytics returns CSV text in raw_result["data"]["raw"].
        Handles:
          • Standard multi-row CSV
          • Single-value aggregate (2-line CSV: header + value)
          • Nested data structures
          • Pre-parsed list-of-dicts
        """
        base: dict = {"columns": [], "rows": [], "error": None}

        if not raw_result.get("success"):
            base["error"] = _extract_error_msg(raw_result.get("error", "Unknown MCP error"))
            return base

        data = raw_result.get("data", {})

        # ── Extract raw CSV text ───────────────────────────────────────────
        raw_text: str | None = None

        if isinstance(data, str):
            raw_text = data
        elif isinstance(data, dict):
            raw_text = data.get("raw")
            if raw_text is None:
                inner = data.get("data", {})
                if isinstance(inner, dict):
                    raw_text = inner.get("raw")

        # ── Parse CSV text ─────────────────────────────────────────────────
        if raw_text and isinstance(raw_text, str) and raw_text.strip():
            rows = self._parse_csv(raw_text.strip())
            if rows:
                base["columns"] = list(rows[0].keys())
                base["rows"]    = rows
                return base

            # 2-line response: header + single aggregate value
            lines = [ln.strip() for ln in raw_text.strip().splitlines() if ln.strip()]
            if len(lines) == 2:
                col = lines[0].strip('"').strip()
                val = lines[1].strip('"').strip()
                base["columns"] = [col]
                base["rows"]    = [{col: val}]
                return base

        # ── Fallback: pre-parsed list-of-dicts ────────────────────────────
        if isinstance(data, list) and data:
            if isinstance(data[0], dict):
                base["columns"] = list(data[0].keys())
                base["rows"]    = data
            else:
                base["columns"] = ["value"]
                base["rows"]    = [{"value": str(v)} for v in data]
            return base

        # ── Fallback: single dict ─────────────────────────────────────────
        if isinstance(data, dict) and data:
            base["columns"] = list(data.keys())
            base["rows"]    = [data]
            return base

        base["error"] = "No parseable data in MCP response"
        return base

    @staticmethod
    def _parse_csv(text: str) -> list[dict]:
        """
        Parse a CSV string into a list of dicts.
        Handles quoted fields, embedded commas, and stray whitespace.
        """
        try:
            reader = csv.DictReader(
                io.StringIO(text),
                skipinitialspace=True,
            )
            rows = []
            for row in reader:
                # Strip whitespace and stray quotes from every key and value
                clean = {
                    _strip_quotes(k): _strip_quotes(v)
                    for k, v in row.items()
                    if k is not None
                }
                rows.append(clean)
            return rows
        except Exception:
            return []

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _empty_result(cid, label, otype, desc, error) -> dict:
        """Kept for backward-compatibility — delegates to module helper."""
        return _empty_result(cid, label, otype, desc, error)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_quotes(value: "str | None") -> str:
    """Remove leading/trailing whitespace and CSV-style quotes."""
    if value is None:
        return ""
    return str(value).strip().strip('"\'')


def _empty_result(
    cid:   str,
    label: str,
    otype: str,
    desc:  str,
    error: str,
) -> dict:
    """Create a QueryResult dict that signals an error or empty data."""
    return {
        "id":          cid,
        "label":       label,
        "output_type": otype,
        "description": desc,
        "columns":     [],
        "rows":        [],
        "error":       error,
    }


def _extract_error_msg(error_val) -> str:
    """
    Extract a clean, human-readable error message from any Zoho MCP error.

    Zoho MCP errors arrive in various shapes:
      • A plain string
      • A dict: {"isError": True, "content": [{"type": "text", "text": "{...json...}"}]}
      • A nested JSON string containing "errorMessage"
    """
    if error_val is None:
        return "Unknown error"

    if isinstance(error_val, str):
        # Try to extract errorMessage from embedded JSON string
        match = re.search(r'"errorMessage"\s*:\s*"([^"]+)"', error_val)
        if match:
            return match.group(1)
        # Try summary
        match = re.search(r'"summary"\s*:\s*"([^"]+)"', error_val)
        if match:
            return match.group(1)
        return error_val[:300]

    if isinstance(error_val, dict):
        # Shape: {isError: True, content: [{type: text, text: '{"errorMessage":"..."}'}]}
        for item in error_val.get("content", []):
            if item.get("type") == "text":
                text = item.get("text", "")
                try:
                    parsed    = json.loads(text)
                    err_msg   = parsed.get("data", {}).get("errorMessage", "")
                    summary   = parsed.get("summary", "")
                    if err_msg:
                        return f"{summary}: {err_msg}" if summary else err_msg
                    if summary:
                        return summary
                except (json.JSONDecodeError, AttributeError):
                    match = re.search(r'"errorMessage"\s*:\s*"([^"]+)"', text)
                    if match:
                        return match.group(1)
        # Direct errorMessage in dict
        if "errorMessage" in error_val:
            return str(error_val["errorMessage"])
        return str(error_val)[:300]

    return str(error_val)[:300]


def _is_permanent_error(error_msg: str) -> bool:
    """
    Return True if the error message indicates a permanent SQL error that
    will never succeed on retry (e.g. INVALID_COLUMN, SYNTAX_ERROR).
    """
    upper = error_msg.upper()
    return any(kw in upper for kw in _PERMANENT_ERROR_KEYWORDS)
