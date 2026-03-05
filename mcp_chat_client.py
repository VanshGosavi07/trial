"""
Zoho CRM Analytics - MCP Chat Client v2.0
Fully working chat interface for Zoho CRM Analytics MCP server
"""

import requests
import json
import time
import re
import sys
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────────
MCP_URL = "https://test-60066497793.zohomcp.in/mcp/message"
API_KEY = "56ee4822d155ac27862d0cf1752ee43d"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}


# ── MCP Client ─────────────────────────────────────────────────────────────────
class ZohoCRMMCPClient:
    def __init__(self):
        self.session = requests.Session()
        self._id = 1
        self.org_id = None
        self.workspace_id = None
        self.workspace_name = None
        self.tables = {}        # lower_name -> {"viewId":..., "viewName":...}
        self.all_views = []
        self.initialized = False

    def _send(self, method, params):
        payload = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}
        self._id += 1
        try:
            r = self.session.post(
                f"{MCP_URL}?key={API_KEY}",
                json=payload, headers=HEADERS, timeout=30
            )
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
        except Exception as e:
            return {"error": str(e)}

    def _call_tool(self, name, arguments):
        resp = self._send("tools/call", {"name": name, "arguments": arguments})
        if "result" not in resp:
            return {"error": resp.get("error", resp)}
        result = resp["result"]
        if result.get("isError"):
            return {"error": result}
        for item in result.get("content", []):
            if item.get("type") == "text":
                try:
                    return json.loads(item["text"])
                except Exception:
                    return {"raw": item["text"]}
        return result

    def initialize(self):
        """Initialize MCP session and auto-discover metadata"""
        print("Connecting to MCP server...")

        # MCP handshake
        resp = self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "zoho-crm-chat", "version": "2.0"}
        })
        if "error" in resp:
            print(f"  ERROR - Initialize failed: {resp['error']}")
            return False

        # Initialized notification
        self.session.post(
            f"{MCP_URL}?key={API_KEY}",
            json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            headers=HEADERS
        )

        # Get org
        print("  Fetching organization...")
        orgs = self._call_tool("ZohoAnalytics_getOrganizations", {})
        if "error" in orgs:
            print(f"  ERROR - Could not get org: {orgs['error']}")
            return False

        org_list = orgs.get("data", {}).get("orgs", [])
        if not org_list:
            print("  ERROR - No organizations found")
            return False

        self.org_id = str(org_list[0]["orgId"])
        org_name = org_list[0].get("orgName", self.org_id)
        print(f"  OK - Org: {org_name} (ID: {self.org_id})")

        # Get workspaces
        print("  Fetching workspaces...")
        ws_data = self._call_tool("ZohoAnalytics_getAllWorkspaces", {
            "headers": {"ZANALYTICS-ORGID": self.org_id}
        })
        owned = ws_data.get("data", {}).get("ownedWorkspaces", [])
        shared = ws_data.get("data", {}).get("sharedWorkspaces", [])
        all_ws = owned + shared

        if not all_ws:
            print("  ERROR - No workspaces found")
            return False

        # Prefer CRM workspace
        ws = next(
            (w for w in all_ws if "crm" in w.get("workspaceName", "").lower()),
            all_ws[0]
        )
        self.workspace_id = str(ws["workspaceId"])
        self.workspace_name = ws.get("workspaceName", self.workspace_id)
        print(f"  OK - Workspace: {self.workspace_name}")

        # Get views
        print("  Loading tables...")
        views_data = self._call_tool("ZohoAnalytics_getViews", {
            "headers": {"ZANALYTICS-ORGID": self.org_id},
            "path_variables": {"workspace-id": self.workspace_id}
        })
        self.all_views = views_data.get("data", {}).get("views", [])

        for v in self.all_views:
            if v.get("viewType") == "Table":
                vname = v["viewName"]
                meta = {"viewId": v["viewId"], "viewName": vname}
                self.tables[vname.lower()] = meta
                self.tables[vname.lower().replace(" ", "_")] = meta

        table_names = [v["viewName"] for v in self.all_views if v.get("viewType") == "Table"]
        print(f"  OK - Found {len(table_names)} tables: {', '.join(table_names)}")

        self.initialized = True
        return True

    def run_sql(self, sql):
        """Execute SQL via async export job. Returns {"success":True, "data":...} or {"error":...}"""
        print(f"\n  Running SQL: {sql}")

        config = json.dumps({"sqlQuery": sql})

        job_resp = self._call_tool("ZohoAnalytics_createExportJobSQLQuery", {
            "headers": {"ZANALYTICS-ORGID": self.org_id},
            "path_variables": {"workspace-id": self.workspace_id},
            "query_params": {"CONFIG": config}
        })

        if "error" in job_resp:
            return {"error": job_resp["error"]}

        job_id = job_resp.get("data", {}).get("jobId") or job_resp.get("jobId")
        if not job_id:
            return {"error": f"No jobId from server. Got: {json.dumps(job_resp)[:300]}"}

        print(f"  Job created (ID: {job_id}) - polling", end="", flush=True)

        for _ in range(30):
            time.sleep(2)
            print(".", end="", flush=True)

            status = self._call_tool("ZohoAnalytics_getExportJobDetails", {
                "headers": {"ZANALYTICS-ORGID": self.org_id},
                "path_variables": {
                    "workspace-id": self.workspace_id,
                    "job-id": str(job_id)
                }
            })

            job_code = str(status.get("data", {}).get("jobCode", ""))

            if job_code == "1004":  # Completed
                print(" done")
                downloaded = self._call_tool("ZohoAnalytics_downloadExportedData", {
                    "headers": {"ZANALYTICS-ORGID": self.org_id},
                    "path_variables": {
                        "workspace-id": self.workspace_id,
                        "job-id": str(job_id)
                    }
                })
                return {"success": True, "data": downloaded}

            elif job_code == "1003":
                print(" failed")
                return {"error": f"Export job failed: {json.dumps(status)[:300]}"}

            elif job_code == "1005":
                print(" not found")
                return {"error": "Job ID not found on server"}

        print(" timeout")
        return {"error": "Job timed out after 60 seconds"}


# ── Natural Language -> SQL ─────────────────────────────────────────────────────
def find_table(client, *keywords):
    """Find a table name by keyword matching"""
    for kw in keywords:
        if kw in client.tables:
            return client.tables[kw]["viewName"]
    for kw in keywords:
        for key, val in client.tables.items():
            if kw in key:
                return val["viewName"]
    return None


def interpret_query(q_orig, client):
    """
    Convert natural language to (sql, description).
    Returns (None, None) if unrecognized.
    """
    q = q_orig.lower().strip()

    # ── Leads ──────────────────────────────────────────────────────────────────
    if re.search(r'\b(leads?)\b', q):
        t = find_table(client, "leads") or "Leads"

        if re.search(r'\b(how many|count|total|number of)\b', q):
            return (f'SELECT COUNT(*) as "Total Leads" FROM "{t}"',
                    "Total Leads Count")

        if re.search(r'\b(by status|per status|lead status)\b', q):
            return (f'SELECT "Lead Status", COUNT(*) as Count FROM "{t}" '
                    f'GROUP BY "Lead Status" ORDER BY Count DESC',
                    "Leads by Status")

        if re.search(r'\b(by source|per source|lead source)\b', q):
            return (f'SELECT "Lead Source", COUNT(*) as Count FROM "{t}" '
                    f'GROUP BY "Lead Source" ORDER BY Count DESC',
                    "Leads by Source")

        if re.search(r'\b(by owner|per owner|owner|assigned)\b', q):
            return (f'SELECT "Lead Owner", COUNT(*) as Count FROM "{t}" '
                    f'GROUP BY "Lead Owner" ORDER BY Count DESC',
                    "Leads by Owner")

        if re.search(r'\b(by industry|industry)\b', q):
            return (f'SELECT "Industry", COUNT(*) as Count FROM "{t}" '
                    f'GROUP BY "Industry" ORDER BY Count DESC',
                    "Leads by Industry")

        if re.search(r'\b(converted)\b', q):
            return (f'SELECT COUNT(*) as "Converted Leads" FROM "{t}" '
                    f'WHERE "Converted" = true',
                    "Converted Leads")

        if re.search(r'\b(today|created today)\b', q):
            d = datetime.now().strftime("%Y-%m-%d")
            return (f'SELECT COUNT(*) as "Leads Today" FROM "{t}" '
                    f'WHERE DATE("Created Time") = \'{d}\'',
                    f"Leads Created Today ({d})")

        if re.search(r'\b(this month|current month)\b', q):
            yr = datetime.now().year
            mo = datetime.now().month
            return (f'SELECT COUNT(*) as "Leads This Month" FROM "{t}" '
                    f'WHERE YEAR("Created Time") = {yr} AND MONTH("Created Time") = {mo}',
                    "Leads Created This Month")

        if re.search(r'\b(recent|latest|last|new)\b', q):
            return (f'SELECT "First Name", "Last Name", "Email", "Lead Status", '
                    f'"Lead Source", "Created Time" FROM "{t}" '
                    f'ORDER BY "Created Time" DESC LIMIT 10',
                    "10 Most Recent Leads")

        if re.search(r'\b(list|show|all|get|display)\b', q):
            return (f'SELECT "First Name", "Last Name", "Email", "Phone", '
                    f'"Lead Status", "Lead Source", "Created Time" FROM "{t}" LIMIT 20',
                    "First 20 Leads")

        # Default: count
        return (f'SELECT COUNT(*) as "Total Leads" FROM "{t}"',
                "Total Leads Count")

    # ── Contacts ───────────────────────────────────────────────────────────────
    if re.search(r'\b(contacts?)\b', q):
        t = find_table(client, "contacts") or "Contacts"

        if re.search(r'\b(how many|count|total|number of)\b', q):
            return (f'SELECT COUNT(*) as "Total Contacts" FROM "{t}"',
                    "Total Contacts Count")

        if re.search(r'\b(by account)\b', q):
            return (f'SELECT "Account Name", COUNT(*) as Count FROM "{t}" '
                    f'GROUP BY "Account Name" ORDER BY Count DESC LIMIT 20',
                    "Contacts by Account")

        return (f'SELECT "First Name", "Last Name", "Email", "Phone", '
                f'"Account Name", "Created Time" FROM "{t}" LIMIT 20',
                "First 20 Contacts")

    # ── Accounts ───────────────────────────────────────────────────────────────
    if re.search(r'\b(accounts?)\b', q):
        t = find_table(client, "accounts") or "Accounts"

        if re.search(r'\b(how many|count|total|number of)\b', q):
            return (f'SELECT COUNT(*) as "Total Accounts" FROM "{t}"',
                    "Total Accounts Count")

        if re.search(r'\b(by industry|industry)\b', q):
            return (f'SELECT "Industry", COUNT(*) as Count FROM "{t}" '
                    f'GROUP BY "Industry" ORDER BY Count DESC',
                    "Accounts by Industry")

        if re.search(r'\b(by type|type)\b', q):
            return (f'SELECT "Account Type", COUNT(*) as Count FROM "{t}" '
                    f'GROUP BY "Account Type" ORDER BY Count DESC',
                    "Accounts by Type")

        return (f'SELECT "Account Name", "Industry", "Account Type", '
                f'"Phone", "Created Time" FROM "{t}" LIMIT 20',
                "First 20 Accounts")

    # ── Deals ──────────────────────────────────────────────────────────────────
    if re.search(r'\b(deals?)\b', q):
        t = find_table(client, "deals") or "Deals"

        if re.search(r'\b(how many|count|total|number of)\b', q):
            return (f'SELECT COUNT(*) as "Total Deals" FROM "{t}"',
                    "Total Deals Count")

        if re.search(r'\bwon\b', q):
            return (f'SELECT COUNT(*) as "Won Deals", SUM("Amount") as "Total Revenue" '
                    f'FROM "{t}" WHERE "Stage" = \'Closed Won\'',
                    "Won Deals & Revenue")

        if re.search(r'\blost\b', q):
            return (f'SELECT COUNT(*) as "Lost Deals" FROM "{t}" '
                    f'WHERE "Stage" = \'Closed Lost\'',
                    "Lost Deals Count")

        if re.search(r'\b(by stage|stage|pipeline)\b', q):
            return (f'SELECT "Stage", COUNT(*) as Count, SUM("Amount") as "Total Amount" '
                    f'FROM "{t}" GROUP BY "Stage" ORDER BY Count DESC',
                    "Deals by Stage")

        if re.search(r'\b(revenue|amount|value|total)\b', q):
            return (f'SELECT SUM("Amount") as "Total Revenue", AVG("Amount") as "Avg Deal Size", '
                    f'COUNT(*) as "Total Deals" FROM "{t}"',
                    "Revenue Summary")

        return (f'SELECT "Deal Name", "Account Name", "Stage", "Amount", '
                f'"Close Date", "Deal Owner" FROM "{t}" '
                f'ORDER BY "Close Date" DESC LIMIT 20',
                "Recent 20 Deals")

    # ── Revenue / Amount (standalone) ──────────────────────────────────────────
    if re.search(r'\b(total revenue|total amount|total value|pipeline value|pipeline)\b', q):
        t = find_table(client, "deals") or "Deals"
        return (f'SELECT SUM("Amount") as "Total Revenue", AVG("Amount") as "Avg Deal Size", '
                f'COUNT(*) as "Total Deals" FROM "{t}"',
                "Revenue Summary")

    # ── Users / Salespeople ────────────────────────────────────────────────────
    if re.search(r'\b(users?|salesperson|salespeople|team|staff)\b', q):
        t = find_table(client, "users") or "Users"
        return (f'SELECT "Full Name", "Email", "Status", "Profile Name", "Role Name" FROM "{t}"',
                "All Users / Sales Team")

    # ── CRM Summary ────────────────────────────────────────────────────────────
    if re.search(r'\b(summary|overview|dashboard|stats|statistics|crm)\b', q):
        lt = find_table(client, "leads") or "Leads"
        ct = find_table(client, "contacts") or "Contacts"
        at = find_table(client, "accounts") or "Accounts"
        dt = find_table(client, "deals") or "Deals"
        return (
            f'SELECT '
            f'(SELECT COUNT(*) FROM "{lt}") as "Total Leads", '
            f'(SELECT COUNT(*) FROM "{ct}") as "Total Contacts", '
            f'(SELECT COUNT(*) FROM "{at}") as "Total Accounts", '
            f'(SELECT COUNT(*) FROM "{dt}") as "Total Deals", '
            f'(SELECT SUM("Amount") FROM "{dt}") as "Total Pipeline Value"',
            "CRM Overview"
        )

    return None, None


# ── Result Formatter ────────────────────────────────────────────────────────────
def parse_csv(text):
    """Parse CSV text (header row + data rows) into list of dicts"""
    import csv, io
    rows = list(csv.reader(io.StringIO(text.strip())))
    if len(rows) < 1:
        return []
    headers = rows[0]
    return [dict(zip(headers, row)) for row in rows[1:]]


def format_result(result_data, description):
    """Render SQL result as a readable table"""
    if not isinstance(result_data, dict):
        return f"Result: {result_data}"

    if "error" in result_data:
        return f"ERROR: {result_data['error']}"

    # Check for raw CSV data (most common from Zoho Analytics export)
    raw = result_data.get("raw")
    if raw and isinstance(raw, str):
        rows = parse_csv(raw)
        if rows:
            return _render_table(rows, description)
        # Single-value result (e.g., "Total Leads\n11")
        lines_of_csv = raw.strip().split("\n")
        if len(lines_of_csv) == 2:
            col, val = lines_of_csv[0].strip(), lines_of_csv[1].strip()
            return f"\n--- {description} ---\n  {col}: {val}"
        return f"\n--- {description} ---\n{raw}"

    # Unwrap nested data
    data = result_data.get("data", result_data)

    rows = None
    if isinstance(data, dict):
        # Check for raw inside data
        if "raw" in data:
            rows = parse_csv(data["raw"]) if isinstance(data["raw"], str) else None
        if rows is None:
            rows = data.get("rows") or data.get("data") or data.get("result")
        if rows is None:
            lines = [f"\n--- {description} ---"]
            for k, v in data.items():
                lines.append(f"  {k}: {v}")
            return "\n".join(lines)
    elif isinstance(data, list):
        rows = data

    if not rows:
        return f"{description}: No data returned"

    return _render_table(rows, description)


def _render_table(rows, description):
    """Render a list of dicts as an ASCII table"""
    if not rows:
        return f"{description}: No rows"

    if not isinstance(rows[0], dict):
        return f"{description}:\n" + "\n".join(str(r) for r in rows[:50])

    cols = list(rows[0].keys())
    col_widths = {}
    for c in cols:
        max_val = max((len(str(row.get(c, ""))) for row in rows), default=0)
        col_widths[c] = min(max(len(str(c)), max_val), 35)

    header = " | ".join(str(c).ljust(col_widths[c]) for c in cols)
    sep = "-+-".join("-" * col_widths[c] for c in cols)

    lines = [f"\n--- {description} ---", "", header, sep]
    for row in rows[:50]:
        line = " | ".join(
            str(row.get(c, "")).ljust(col_widths[c])[:col_widths[c]] for c in cols
        )
        lines.append(line)

    if len(rows) > 50:
        lines.append(f"  ... {len(rows) - 50} more rows not shown")

    lines.append(f"\n  {len(rows)} row(s) returned")
    return "\n".join(lines)


# ── Helpers ─────────────────────────────────────────────────────────────────────
def list_tables(client):
    tables = [v for v in client.all_views if v.get("viewType") == "Table"]
    reports = [v for v in client.all_views if v.get("viewType") == "AnalysisView"]
    pivots = [v for v in client.all_views if v.get("viewType") == "Pivot"]
    dashboards = [v for v in client.all_views if v.get("viewType") == "Dashboard"]

    lines = [f"\n=== VIEWS IN '{client.workspace_name}' ==="]
    lines.append(f"\nTABLES ({len(tables)}) - use these in SQL queries:")
    for v in tables:
        lines.append(f'  "{v["viewName"]}"')
    lines.append(f"\nREPORTS ({len(reports)}):")
    for v in reports[:15]:
        lines.append(f'  {v["viewName"]}')
    if len(reports) > 15:
        lines.append(f"  ... and {len(reports)-15} more")
    lines.append(f"\nPIVOTS ({len(pivots)}), DASHBOARDS ({len(dashboards)})")
    return "\n".join(lines)


HELP_TEXT = """
=== ZOHO CRM ANALYTICS - CHAT ===

LEADS:
  how many leads are created
  leads by status / by source / by owner / by industry
  show recent leads / list all leads
  how many leads converted
  leads created today / this month

CONTACTS:
  how many contacts
  list contacts / contacts by account

ACCOUNTS:
  how many accounts
  accounts by industry / by type
  list accounts

DEALS:
  how many deals
  won deals / lost deals
  deals by stage / deals pipeline
  total revenue / deal value

GENERAL:
  crm summary          (counts for all modules)
  list users / team

COMMANDS:
  sql: SELECT ...      (run any custom SQL)
  tables               (list all tables and views)
  help                 (show this menu)
  exit / quit
"""


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ZOHO CRM ANALYTICS - MCP CHAT CLIENT v2.0")
    print("=" * 60)

    client = ZohoCRMMCPClient()

    if not client.initialize():
        print("\nFailed to connect. Exiting.")
        sys.exit(1)

    print("\nReady! Type 'help' for examples, 'exit' to quit.\n")

    history = []

    while True:
        try:
            raw = input("You: ").strip()
            if not raw:
                continue

            cmd = raw.lower().strip()

            # Special commands
            if cmd in ("exit", "quit", "bye"):
                print("Goodbye!")
                break

            if cmd == "help":
                print(HELP_TEXT)
                continue

            if cmd in ("tables", "views"):
                print(list_tables(client))
                continue

            if cmd == "history":
                if not history:
                    print("  (no history)")
                else:
                    for i, h in enumerate(history, 1):
                        print(f"\n[{i}] You: {h['q']}")
                        print(f"     Bot: {h['a'][:150]}...")
                continue

            # Raw SQL
            if cmd.startswith("sql:"):
                sql = raw[4:].strip()
                if not sql:
                    print('  Usage: sql: SELECT COUNT(*) FROM "Leads"')
                    continue
                res = client.run_sql(sql)
                if res.get("success"):
                    answer = format_result(res["data"], f"SQL Result")
                else:
                    answer = f"SQL Error: {res.get('error')}"
                print(f"\nBot: {answer}\n")
                history.append({"q": raw, "a": answer})
                continue

            # Natural language
            sql, description = interpret_query(raw, client)

            if sql:
                res = client.run_sql(sql)
                if res.get("success"):
                    answer = format_result(res["data"], description)
                else:
                    answer = f"Query failed: {res.get('error')}"
            else:
                answer = (
                    "I couldn't understand that. Try:\n"
                    "  'how many leads' / 'leads by status' / 'crm summary'\n"
                    "  'how many deals' / 'won deals' / 'total revenue'\n"
                    "  'sql: SELECT COUNT(*) FROM \"Leads\"'\n"
                    "  Type 'help' for all options."
                )

            print(f"\nBot: {answer}\n")
            history.append({"q": raw, "a": answer})

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
