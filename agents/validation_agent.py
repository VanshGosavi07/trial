"""
Agent 4: Data Validation Agent
────────────────────────────────
Sits between MCPDataAgent and VizAgent in the pipeline.

Responsibilities:
  1. Validate SQL safety before execution (blocks DDL / DML).
  2. Sanitize every row value returned from Zoho Analytics:
       • Strip whitespace and stray quotes
       • Coerce string numbers → int / float (handles "$1,234", "45%", "1.5K")
       • Normalise NULL / empty → Python None
  3. Detect data quality issues and emit warnings (all-zeros, all-null
     columns, no numeric column for a chart type) so the orchestrator
     can surface them without crashing.

No external API calls — pure Python, fast.
"""

import re
from datetime import datetime
from typing import Any


# ─── Public types ─────────────────────────────────────────────────────────────
QueryResult  = dict   # as returned by MCPDataAgent
QueryPlan    = dict   # as returned by QueryAgent


class ValidationAgent:
    """
    Data Validation & Sanitization Agent.
    Call :meth:`validate_plan` before MCP execution.
    Call :meth:`validate_results` after MCP execution.
    """

    # Characters that are NOT part of a number (preserves - + . e E digits)
    _NON_NUMERIC = re.compile(r"[^\d.\-+eE]")

    # SQL safety guards
    _DANGEROUS_KW = re.compile(
        r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|EXEC|CALL)\b",
        re.IGNORECASE,
    )
    _MUST_START_SELECT = re.compile(r"^\s*SELECT\b", re.IGNORECASE)

    # K / M / B numeric suffixes
    _SUFFIXES = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}

    # Chart types that require at least one numeric column
    _CHART_NEEDS_NUMERIC = {
        "bar", "column", "horizontalBar", "stacked", "stackedbar", "groupedBar",
        "butterfly", "waterfall", "histogram",
        "line", "area", "multiLine", "combination", "combo",
        "scatter", "bubble", "packedbubble",
        "pie", "doughnut", "donut", "ring", "halfpie", "halfring", "halfdoughnut",
        "sunburst", "treemap", "funnel", "heatmap", "radar", "spider", "web",
        "polarArea", "box", "violin", "gauge", "dial", "bullet",
    }

    _OUTPUT_TYPE_ALIASES = {
        "barr": "bar",
        "bars": "bar",
        "horizontalbar": "horizontalBar",
        "hbar": "horizontalBar",
        "groupedbar": "groupedBar",
        "groupbar": "groupedBar",
        "multiline": "multiLine",
        "polar": "polarArea",
        "polarea": "polarArea",
        "donught": "doughnut",
        "dougnut": "doughnut",
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Plan Validation (pre-execution)
    # ─────────────────────────────────────────────────────────────────────────

    def validate_plan(self, plan: QueryPlan) -> QueryPlan:
        """
        Inspect every SQL component in *plan*.
        - Blocks non-SELECT statements (replaces SQL with safe no-op).
        - Blocks any SQL containing dangerous keywords.
        - Returns the (possibly modified) plan — never raises.
        """
        for comp in plan.get("components", []):
            sql = (comp.get("sql") or "").strip()
            cid = comp.get("id", "?")

            if not sql:
                continue

            # Must start with SELECT
            if not self._MUST_START_SELECT.match(sql):
                print(f"[ValidationAgent] ⛔ Component '{cid}': non-SELECT SQL blocked.")
                comp["sql"] = "SELECT 1 AS blocked WHERE 1=0"
                comp["_blocked"] = True
                continue

            # Block dangerous keywords
            if self._DANGEROUS_KW.search(sql):
                print(f"[ValidationAgent] ⛔ Component '{cid}': dangerous keyword blocked.")
                comp["sql"] = "SELECT 1 AS blocked WHERE 1=0"
                comp["_blocked"] = True
                continue

        return plan

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Result Validation (post-execution)
    # ─────────────────────────────────────────────────────────────────────────

    def validate_results(self, results: list[QueryResult]) -> list[QueryResult]:
        """
        Sanitize and quality-check each QueryResult.
        Returns a new list with sanitized rows; original is not modified.
        """
        cleaned: list[QueryResult] = []
        for qr in results:
            if qr.get("error"):
                cleaned.append(qr)
                continue

            rows  = qr.get("rows",    [])
            cols  = qr.get("columns", [])
            otype = self._normalize_output_type(qr.get("output_type", "bar"))

            # Sanitize every row
            sanitized = [self._sanitize_row(r) for r in rows]

            # Re-derive columns from sanitized data if missing
            if sanitized and not cols:
                cols = list(sanitized[0].keys())
            elif sanitized and cols:
                # Re-key using sanitized keys (stripped)
                cols = [str(c).strip().strip('"\'') for c in cols]

            # Quality checks
            issues = self._quality_check(sanitized, cols, otype)
            if issues:
                lbl = qr.get("label", qr.get("id", "?"))
                for issue in issues:
                    print(f"[ValidationAgent] ⚠️  '{lbl}': {issue}")

            cleaned.append({
                **qr,
                "output_type":      otype,
                "rows":             sanitized,
                "columns":          cols,
                "_quality_issues":  issues or None,
            })

        return cleaned

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _sanitize_row(self, row: dict) -> dict:
        """Return a new dict with cleaned keys and coerced values."""
        clean: dict[str, Any] = {}
        for raw_key, raw_val in row.items():
            key = str(raw_key).strip().strip('"\'')

            if raw_val is None:
                clean[key] = None
                continue

            sv = str(raw_val).strip()

            # Treat common NULL-like strings as None
            if sv.lower() in ("", "null", "none", "n/a", "na", "-", "--"):
                clean[key] = None
                continue

            # Date/time columns from Zoho often arrive as compact strings
            # like "250220261106" (ddmmyyyyHHMM). Format them first and skip
            # numeric coercion so leading zeros are not lost.
            dt = self._coerce_datetime_like(key, sv)
            if dt is not None:
                clean[key] = dt
                continue

            # Try numeric coercion
            num = self._coerce_numeric(sv)
            clean[key] = num if num is not None else sv

        return clean

    def _coerce_numeric(self, value: str) -> "float | int | None":
        """
        Convert a string to int / float if possible.
        Handles: "1,234"  →  1234
                 "$1,234.56" → 1234.56
                 "45%"       → 45.0
                 "1.5K"      → 1500
                 "2.3M"      → 2300000
        Returns None if the value is not numeric.
        """
        v = value.strip()
        if not v:
            return None

        # Percentage → strip '%'
        if v.endswith("%"):
            try:
                return float(v[:-1])
            except ValueError:
                pass

        # K / M / B suffix
        upper = v.upper()
        for suffix, mult in self._SUFFIXES.items():
            if upper.endswith(suffix):
                try:
                    result = float(upper[:-1].replace(",", "")) * mult
                    return int(result) if result == int(result) else result
                except ValueError:
                    pass

        # Strip everything except digits, dot, minus, plus, e/E
        leading_minus = v.startswith("-")
        stripped = self._NON_NUMERIC.sub("", v)
        if leading_minus and not stripped.startswith("-"):
            stripped = "-" + stripped

        if not stripped or stripped in ("-", "+", "."):
            return None

        try:
            f = float(stripped)
            # Return int if lossless
            return int(f) if f == int(f) else f
        except (ValueError, OverflowError):
            return None

    def _coerce_datetime_like(self, key: str, value: str) -> str | None:
        """
        Parse compact Zoho datetime tokens for date/time fields.

        Supported forms:
          - ddmmyyyyHHMM       (12 digits)
          - ddmmyyyyHHMMSS     (14 digits)
          - yyyymmdd           (8 digits)
          - numeric token missing one leading zero (11/13 digits)
        """
        key_l = (key or "").lower()
        if not any(tok in key_l for tok in ("time", "date")):
            return None

        token = value.strip()
        if not token:
            return None

        # Keep already-readable values as-is.
        if any(sep in token for sep in ("-", "/", ":", " ")):
            return token

        # Compact numeric datetime/date tokens
        if not re.fullmatch(r"\d{8,14}", token):
            return None

        # Leading zero may have been dropped by upstream conversion.
        if len(token) in (11, 13):
            token = "0" + token

        try:
            if len(token) == 14:
                dt = datetime.strptime(token, "%d%m%Y%H%M%S")
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            if len(token) == 12:
                dt = datetime.strptime(token, "%d%m%Y%H%M")
                return dt.strftime("%Y-%m-%d %H:%M")
            if len(token) == 8:
                # Prefer yyyymmdd for date-only fields in analytics exports.
                dt = datetime.strptime(token, "%Y%m%d")
                return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

        return None

    # ─────────────────────────────────────────────────────────────────────────

    def _quality_check(
        self,
        rows: list[dict],
        cols: list[str],
        otype: str,
    ) -> list[str]:
        """
        Return a list of warning strings for data quality issues.
        Empty list means the data looks clean.
        """
        warnings: list[str] = []

        if not rows:
            warnings.append("Query returned 0 rows")
            return warnings

        if not cols:
            warnings.append("No columns in result set")
            return warnings

        # For chart types that need numeric data, ensure at least one numeric column
        if otype in self._CHART_NEEDS_NUMERIC:
            has_numeric = any(
                self._col_is_numeric(col, rows) for col in cols
            )
            if not has_numeric:
                warnings.append(
                    f"No numeric column found for '{otype}' chart — "
                    "chart may render empty or fall back to a table."
                )

        # Warn about all-zero numeric columns (likely a data or SQL issue)
        for col in cols:
            numeric_vals = [
                r[col] for r in rows
                if isinstance(r.get(col), (int, float))
            ]
            if numeric_vals and all(v == 0 for v in numeric_vals):
                warnings.append(
                    f"Column '{col}' contains all zeros — "
                    "verify the SQL aggregation is correct."
                )

        # Warn about heavily-null columns (>80 % null)
        for col in cols:
            null_count = sum(1 for r in rows if r.get(col) is None)
            if len(rows) >= 5 and null_count / len(rows) > 0.8:
                warnings.append(
                    f"Column '{col}' is {null_count}/{len(rows)} null — "
                    "check column name or table join."
                )

        return warnings

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _col_is_numeric(col: str, rows: list[dict]) -> bool:
        """Return True if the majority of non-null values in *col* are numeric."""
        sample = [r[col] for r in rows[:20] if r.get(col) is not None]
        if not sample:
            return False
        numeric = sum(1 for v in sample if isinstance(v, (int, float)))
        return numeric > len(sample) / 2

    @classmethod
    def _normalize_output_type(cls, otype: str | None) -> str:
        if not otype:
            return "bar"
        txt = str(otype).strip()
        if txt in cls._CHART_NEEDS_NUMERIC:
            return txt

        low = txt.lower()
        if low in cls._CHART_NEEDS_NUMERIC:
            return low

        canonical = re.sub(r"[^a-z0-9]", "", low)
        if canonical in cls._OUTPUT_TYPE_ALIASES:
            return cls._OUTPUT_TYPE_ALIASES[canonical]

        return txt if txt else "bar"
