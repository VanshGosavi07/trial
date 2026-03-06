"""
Agent Orchestrator
──────────────────
Central coordinator for the multi-agent pipeline:

  User Prompt
      │
      ▼
  QueryAgent       ── understands intent, generates SQL QueryPlan
      │
      ▼
  ValidationAgent  ── validates SQL safety (pre-execution)
      │
      ▼
  MCPDataAgent     ── executes SQL against Zoho Analytics
      │
      ▼
  ValidationAgent  ── sanitizes & quality-checks returned data
      │
      ▼
  VizAgent         ── transforms clean data → Plotly widget specs
      │
      ▼
  Final JSON Response → Flask → Frontend

Response payload shape:
{
  "success":   true,
  "message":   "<prose summary for chat bubble>",
  "dashboard": {               ← present only for chart/dashboard results
    "title":      "…",
    "dashboard":  true,
    "components": [ <WidgetSpec>, … ]
  }
}
"""

import time
import re
from .query_agent      import QueryAgent
from .mcp_agent        import MCPDataAgent
from .viz_agent        import VizAgent
from .validation_agent import ValidationAgent


class AgentOrchestrator:
    """
    Initialise once; call .process() per user request.
    Thread-safe for the single-threaded HTTPServer; wrap in a lock if moving to
    multi-thread/async.
    """

    def __init__(self):
        print("[Orchestrator] Initialising agents…")
        self.query_agent      = QueryAgent()
        self.mcp_agent        = MCPDataAgent()
        self.viz_agent        = VizAgent()
        self.validation_agent = ValidationAgent()

        # Pre-connect to MCP so the first user request is served fast
        self.mcp_agent.ensure_connected()
        self._prefetch_schemas()
        print("[Orchestrator] Ready.")

    # ─────────────────────────────────────────────────────────────────────────
    def _prefetch_schemas(self) -> None:
        """Warm the schema cache at startup so QueryAgent always has real column names."""
        tables = self.mcp_agent.available_tables
        if not tables:
            print("[Orchestrator] No tables to prefetch schemas for — skipping.")
            return
        print(f"[Orchestrator] Prefetching schemas for {len(tables)} table(s)…")
        try:
            self.mcp_agent.get_table_schemas(tables)
            print(f"[Orchestrator] Schemas cached: {list(self.mcp_agent._schema_cache.keys())}")
        except Exception as exc:
            print(f"[Orchestrator] Schema prefetch failed (continuing without): {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    def process(self, user_prompt: str, history: list[dict] | None = None) -> dict:
        """
        Full pipeline: prompt → JSON response.

        Returns:
        {
          "success":   bool,
          "message":   str,
          "dashboard": dict | None,
          "error":     str   (only on failure)
        }
        """
        t0 = time.time()
        print(f"\n[Orchestrator] ─── Processing: '{user_prompt[:80]}' ───")

        history = history or []

        # ── 1. Ensure MCP is connected ─────────────────────────────────────
        if not self.mcp_agent.ensure_connected():
            return {
                "success": False,
                "error":   "Could not connect to Zoho Analytics MCP server.",
                "message": "⚠️ Unable to connect to Zoho Analytics. Please refresh the page or check your MCP connection."
            }

        available_tables = self.mcp_agent.available_tables
        if not available_tables:
            return {
                "success": False,
                "error":   "No tables found in workspace.",
                "message": "⚠️ No tables found in your Zoho Analytics workspace."
            }

        # ── 2. Query Agent: understand intent → SQL plan ───────────────────
        print(f"[Orchestrator] Step 1 ─ QueryAgent: building plan…")
        try:
            query_plan = self.query_agent.build_plan(
                user_prompt, available_tables, history,
                table_schemas=self.mcp_agent._schema_cache or None,
            )
        except Exception as e:
            print(f"[Orchestrator] QueryAgent failed: {e}")
            query_plan = self.query_agent._fallback_plan(user_prompt, available_tables)

        intent = query_plan.get("intent", "dashboard")
        title  = query_plan.get("title",  "Analytics")
        n_comp = len(query_plan.get("components", []))
        print(f"[Orchestrator] Plan: intent={intent}, title={title}, components={n_comp}")

        # ── 3. Validation Agent: validate SQL safety ───────────────────────
        print(f"[Orchestrator] Step 2 ─ ValidationAgent: checking SQL safety…")
        query_plan = self.validation_agent.validate_plan(query_plan)

        # ── 4. MCP Agent: execute SQL ──────────────────────────────────────
        print(f"[Orchestrator] Step 3 ─ MCPDataAgent: executing {n_comp} quer{'y' if n_comp==1 else 'ies'}…")
        query_results = self.mcp_agent.execute_plan(query_plan)

        # ── 5. Validation Agent: sanitize & quality-check results ──────────
        print(f"[Orchestrator] Step 4 ─ ValidationAgent: sanitizing results…")
        query_results = self.validation_agent.validate_results(query_results)

        # Separate successful vs errored results
        good_results  = [r for r in query_results if not r.get("error")]
        error_results = [r for r in query_results if     r.get("error")]

        if error_results and not good_results:
            errors_str = "; ".join(r["error"] for r in error_results[:3])
            return {
                "success": False,
                "error":   errors_str,
                "message": f"⚠️ Data fetch failed: {errors_str}"
            }

        # ── 6. VizAgent: data → widget specs ──────────────────────────────
        print(f"[Orchestrator] Step 5 ─ VizAgent: building {len(good_results)} widget(s)…")
        widgets = self.viz_agent.build_widgets(good_results, user_prompt)

        # Log failed components but do NOT render them on the dashboard
        for er in error_results:
            print(
                f"[Orchestrator] ✗ Excluded from dashboard: "
                f"'{er.get('label', '?')}' — {er.get('error', 'unknown error')}"
            )

        # Attach data quality warnings to widgets that have them
        for qr, widget in zip(good_results, widgets):
            issues = qr.get("_quality_issues")
            if issues:
                widget["_warnings"] = issues

        # ── 7. Build prose summary ─────────────────────────────────────────
        prose = self._build_prose(intent, title, query_results, widgets, user_prompt)

        # ── 8. Build response payload ──────────────────────────────────────
        elapsed = round(time.time() - t0, 2)
        print(f"[Orchestrator] Done in {elapsed}s ─ {len(widgets)} widget(s)")

        # Single numeric answer: keep text-only for plain Q&A, but return a
        # KPI widget when user explicitly asks for KPI/card/widget or add/append.
        if intent == "numeric" and len(widgets) == 1:
            explicit_widget_request = bool(re.search(
                r"\b(kpi|metric|card|widget|dashboard|chart|graph|add|append|include|insert)\b",
                user_prompt or "",
                re.IGNORECASE,
            ))
            if not explicit_widget_request:
                return {"success": True, "message": prose, "dashboard": None}

        # Everything else → prose + dashboard JSON
        dashboard_json = {
            "dashboard":  True,
            "title":      title,
            "components": widgets
        }

        return {
            "success":   True,
            "message":   self._format_final_message(prose, dashboard_json),
            "dashboard": dashboard_json
        }

    # ─────────────────────────────────────────────────────────────────────────
    def _build_prose(self, intent, title, results, widgets, user_prompt) -> str:
        """
        Build a concise markdown prose summary for the chat bubble.
        We do this deterministically from the data — fast, no extra API call
        for standard cases.
        """
        lines = []

        kpi_widgets = [w for w in widgets if w.get("chart_type") in ("kpi", "numeric")]
        viz_widgets = [w for w in widgets if w.get("chart_type") not in ("kpi", "numeric")]

        if intent == "numeric":
            # Single number answer
            if kpi_widgets:
                d   = kpi_widgets[0].get("data", {})
                val = d.get("value", "—")
                sub = d.get("subtitle", kpi_widgets[0].get("title", ""))
                return f"**{val}** {sub}"

        # Summary line for dashboards
        if kpi_widgets:
            kpi_parts = []
            for w in kpi_widgets:
                d   = w.get("data", {})
                val = d.get("value", "—")
                lbl = w.get("title", "")
                kpi_parts.append(f"**{val}** {lbl}")
            lines.append("  ·  ".join(kpi_parts))

        if viz_widgets:
            chart_names = [w.get("title", "") for w in viz_widgets[:4]]
            lines.append(f"Charts: {', '.join(chart_names)}")

        if not lines:
            return f"Your **{title}** is ready."

        return "\n\n".join(lines)

    def _format_final_message(self, prose: str, dashboard: dict) -> str:
        """
        Combine prose summary with the dashboard JSON code-block so the
        existing frontend extractVisualization() function picks it up.
        """
        import json
        dash_json = json.dumps(dashboard, ensure_ascii=False)
        return f"{prose}\n\n```json\n{dash_json}\n```"

    # ─────────────────────────────────────────────────────────────────────────
    def workspace_info(self) -> dict:
        return {
            "workspace":       self.mcp_agent.workspace_name,
            "available_tables": self.mcp_agent.available_tables,
            "connected":       self.mcp_agent._mcp is not None
                               and self.mcp_agent._mcp.initialized
        }
