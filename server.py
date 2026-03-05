"""
Zoho CRM Chat - Local Proxy Server
Serves the HTML page and proxies requests to the MCP server (bypasses CORS)
Run: python server.py  then open http://localhost:8000
"""

import json
import sys
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import time
import re
import os

# ── Config ─────────────────────────────────────────────────────────────────────
PORT = 8000
MCP_URL = "https://test-60066497793.zohomcp.in/mcp/message"
API_KEY  = "56ee4822d155ac27862d0cf1752ee43d"
BASE_DIR = Path(__file__).parent

# ── Re-use MCP client logic ────────────────────────────────────────────────────
sys.path.insert(0, str(BASE_DIR))
from mcp_chat_client import ZohoCRMMCPClient, interpret_query, format_result

# Global MCP client (shared across requests)
mcp_client = None

def get_client():
    global mcp_client
    if mcp_client is None or not mcp_client.initialized:
        mcp_client = ZohoCRMMCPClient()
        mcp_client.initialize()
    return mcp_client


# ── HTTP Request Handler ───────────────────────────────────────────────────────
class ChatHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print(f"  [{self.address_string()}] {format % args}")

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/" or path == "/index.html":
            self._serve_file(BASE_DIR / "index.html", "text/html; charset=utf-8")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.startswith("/chat"):
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                data = json.loads(body)
                user_msg = data.get("message", "").strip()

                if not user_msg:
                    self._json_response({"error": "Empty message"}, 400)
                    return

                client = get_client()
                response = self._handle_query(user_msg, client)
                self._json_response({"reply": response})

            except Exception as e:
                self._json_response({"error": str(e)}, 500)
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_query(self, user_msg, client):
        """Process user query and return formatted response"""
        cmd = user_msg.lower().strip()

        if cmd in ("help", "?"):
            return (
                "I can answer questions about your Zoho CRM data. Try:\n\n"
                "• how many leads are in the crm\n"
                "• leads by status / by source / by owner\n"
                "• show recent leads\n"
                "• how many contacts / accounts / deals\n"
                "• deals by stage / won deals / total revenue\n"
                "• crm summary\n"
                "• tables (list all tables)\n"
                "• sql: SELECT COUNT(*) FROM \"Leads\""
            )

        if cmd == "tables":
            tables = [v["viewName"] for v in client.all_views if v.get("viewType") == "Table"]
            return "Available tables:\n" + "\n".join(f"• {t}" for t in tables)

        # Raw SQL
        if cmd.startswith("sql:"):
            sql = user_msg[4:].strip()
            if not sql:
                return 'Usage: sql: SELECT COUNT(*) FROM "Leads"'
            res = client.run_sql(sql)
            if res.get("success"):
                return format_result(res["data"], "SQL Result")
            return f"SQL Error: {res.get('error')}"

        # Natural language
        sql, description = interpret_query(user_msg, client)
        if sql:
            res = client.run_sql(sql)
            if res.get("success"):
                return format_result(res["data"], description)
            return f"Query failed: {res.get('error')}"

        return (
            "I couldn't understand that. Try:\n\n"
            "• how many leads\n"
            "• leads by status\n"
            "• crm summary\n"
            "• Type 'help' for all options"
        )

    def _serve_file(self, filepath, content_type):
        try:
            content = filepath.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(content))
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()

    def _json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(body)


# ── Entry Point ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ZOHO CRM ANALYTICS - WEB CHAT SERVER")
    print("=" * 60)
    print(f"\nInitializing MCP connection...")
    get_client()  # Pre-warm connection
    print(f"\nServer running at: http://localhost:{PORT}")
    print(f"Open your browser and go to: http://localhost:{PORT}")
    print(f"\nPress Ctrl+C to stop\n")

    server = HTTPServer(("", PORT), ChatHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
