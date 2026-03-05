"""
Analytics AI — Flask Server
────────────────────────────
Multi-agent pipeline for interactive analytics dashboards.

Endpoints:
  GET  /              → frontend (index_new.html)
  GET  /api/status    → workspace connection health check
  POST /api/chat      → main agent pipeline
  GET  /api/tables    → list available workspace tables
  GET  /api/health    → lightweight liveness probe
  GET  /<path>        → static file passthrough

Run:  python server.py
Open: http://localhost:8000
"""

import json
import os
import sys
import traceback
from pathlib import Path

from flask import Flask, jsonify, redirect, request, send_from_directory
from flask_cors import CORS

# ── Path bootstrap ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))


def _load_dotenv(dotenv_path: Path) -> None:
    """Tiny .env loader (no extra dependency)."""
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(BASE_DIR / ".env")

from agents.orchestrator import AgentOrchestrator

# ── Flask application ──────────────────────────────────────────────────────────
app  = Flask(__name__, static_folder=str(BASE_DIR))
CORS(app)

PORT = int(os.environ.get("PORT", 8000))

# ── Global orchestrator — single instance initialised at startup ───────────────
print("=" * 60)
print("  ANALYTICS AI — MULTI-AGENT DASHBOARD SERVER")
print("=" * 60)
orchestrator = AgentOrchestrator()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main frontend."""
    return send_from_directory(str(BASE_DIR), "index_new.html")


@app.route("/api/health")
def health():
    """Lightweight liveness probe — always returns 200."""
    return jsonify({"status": "ok"})


@app.route("/api/status")
def status():
    """Connection / authentication health check."""
    info = orchestrator.workspace_info()
    return jsonify({
        "authenticated": info["connected"],
        "workspace":     info["workspace"],
        "tables":        info["available_tables"],
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main agent endpoint.

    Request body (JSON):
      { "message": "...", "history": [...] }

    Response (JSON):
      { "success": true, "message": "...", "dashboard": {...} | null }
    """
    body     = request.get_json(silent=True) or {}
    user_msg = (body.get("message") or "").strip()
    history  = body.get("history") or []

    if not user_msg:
        return jsonify({"success": False, "error": "Empty message"}), 400

    # Validate history is a list of dicts
    if not isinstance(history, list):
        history = []

    print(f"\n[API] /api/chat  message={user_msg[:80]!r}")

    try:
        result = orchestrator.process(user_msg, history)
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error":   str(exc),
            "message": f"Internal server error: {exc}",
        }), 500


@app.route("/api/tables")
def tables():
    """List available tables in the Zoho Analytics workspace."""
    info = orchestrator.workspace_info()
    return jsonify({
        "success":   True,
        "workspace": info["workspace"],
        "tables":    info["available_tables"],
    })


@app.route("/oauth/authorize")
def oauth_redirect():
    """Placeholder — OAuth not required for MCP key auth."""
    return redirect("/")


@app.route("/<path:filename>")
def static_files(filename):
    """Serve static assets (JS, CSS, images) from the project root."""
    return send_from_directory(str(BASE_DIR), filename)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n  Server running at: http://localhost:{PORT}")
    print("  Open in your browser to start.\n")
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
