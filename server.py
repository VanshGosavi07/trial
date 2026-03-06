"""
Analytics AI — Flask Server
────────────────────────────
Multi-agent pipeline for interactive analytics dashboards.

Endpoints:
  GET  /              → frontend (index_new.html)
  GET  /api/status    → workspace connection health check
  POST /api/chat      → main agent pipeline
    POST /api/n8n/chat  → n8n-friendly chat endpoint (same contract)
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
from datetime import datetime, timezone
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _load_n8n_settings() -> dict:
    return {
        "enabled": _env_bool("N8N_ENABLED", False),
        "webhook_url": (os.environ.get("N8N_WEBHOOK_URL") or "").strip(),
        "mode": (os.environ.get("N8N_MODE") or "mirror").strip().lower(),
        "timeout_sec": float(os.environ.get("N8N_TIMEOUT_SEC", "12")),
        "secret": (os.environ.get("N8N_SHARED_SECRET") or "").strip(),
    }


N8N = _load_n8n_settings()


def _n8n_is_enabled() -> bool:
    return bool(N8N["enabled"] and N8N["webhook_url"])


def _post_to_n8n(payload: dict) -> dict | None:
    """POST JSON payload to n8n webhook and return parsed JSON or None."""
    if not _n8n_is_enabled():
        return None

    headers = {"Content-Type": "application/json"}
    if N8N["secret"]:
        headers["X-Shared-Secret"] = N8N["secret"]

    req = urlrequest.Request(
        N8N["webhook_url"],
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urlrequest.urlopen(req, timeout=N8N["timeout_sec"]) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if not raw.strip():
                return {}
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {"data": parsed}
            except json.JSONDecodeError:
                return {"raw": raw}
    except (urlerror.URLError, TimeoutError) as exc:
        print(f"[n8n] Webhook call failed: {exc}")
        return None


def _is_chat_result(payload: dict | None) -> bool:
    if not isinstance(payload, dict):
        return False
    if "success" not in payload:
        return False
    return ("message" in payload) or ("dashboard" in payload)


def _run_local_orchestrator(user_msg: str, history: list) -> dict:
    return orchestrator.process(user_msg, history)

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
        "n8n": {
            "enabled": _n8n_is_enabled(),
            "mode": N8N["mode"],
        },
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
        request_event = {
            "event": "chat.request",
            "source": "analytics-ai",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": user_msg,
            "history": history,
        }

        # Proxy mode: n8n gets first chance to return a final app response.
        if _n8n_is_enabled() and N8N["mode"] == "proxy":
            n8n_reply = _post_to_n8n(request_event)
            if _is_chat_result(n8n_reply):
                n8n_reply["meta"] = {
                    **(n8n_reply.get("meta") or {}),
                    "handled_by": "n8n",
                    "mode": "proxy",
                }
                return jsonify(n8n_reply)
            if isinstance(n8n_reply, dict) and _is_chat_result(n8n_reply.get("result")):
                out = n8n_reply["result"]
                out["meta"] = {
                    **(out.get("meta") or {}),
                    "handled_by": "n8n",
                    "mode": "proxy",
                }
                return jsonify(out)

            print("[n8n] Proxy mode fallback to local orchestrator.")

        result = _run_local_orchestrator(user_msg, history)

        # Mirror mode: send request + local result to n8n for logging/automation.
        if _n8n_is_enabled() and N8N["mode"] in {"mirror", "proxy"}:
            _post_to_n8n({
                **request_event,
                "event": "chat.response",
                "result": result,
            })

        result["meta"] = {
            **(result.get("meta") or {}),
            "handled_by": "local-orchestrator",
            "n8n_enabled": _n8n_is_enabled(),
            "n8n_mode": N8N["mode"],
        }
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error":   str(exc),
            "message": f"Internal server error: {exc}",
        }), 500


@app.route("/api/n8n/chat", methods=["POST"])
def n8n_chat():
    """
    n8n-facing endpoint with same payload contract as /api/chat.
    Useful when n8n orchestrates upstream logic and invokes this app for planning.
    """
    body = request.get_json(silent=True) or {}
    user_msg = (body.get("message") or "").strip()
    history = body.get("history") or []

    if not user_msg:
        return jsonify({"success": False, "error": "Empty message"}), 400
    if not isinstance(history, list):
        history = []

    shared_secret = (N8N.get("secret") or "").strip()
    if shared_secret:
        provided = (request.headers.get("X-Shared-Secret") or "").strip()
        if provided != shared_secret:
            return jsonify({"success": False, "error": "Unauthorized"}), 401

    try:
        result = _run_local_orchestrator(user_msg, history)
        result["meta"] = {
            **(result.get("meta") or {}),
            "endpoint": "n8n-chat",
            "handled_by": "local-orchestrator",
        }
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(exc),
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
