"""
Agent 1: Query Understanding Agent
─────────────────────────────────
Receives the raw user prompt + available table names.
Uses Groq (openai/gpt-oss-120b) to:
  • Understand the intent (dashboard / chart / kpi / table / numeric / summary)
  • Identify which tables & fields are needed
  • Generate precise SQL queries for each required component
  • Return a structured QueryPlan
"""

import json
import os
import re
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


class QueryAgent:
    """
    Query Understanding Agent.
    Input : user prompt (str) + available_tables (list[str]) + conversation_history
    Output: QueryPlan dict
    """

    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    # ─────────────────────────────────────────────────────────────────────────
    def build_plan(
        self,
        user_prompt:      str,
        available_tables: list[str],
        history:          list[dict] | None = None,
        table_schemas:    dict[str, list[str]] | None = None,
    ) -> dict:
        """
        Build and return a QueryPlan dict.

        Parameters
        ----------
        user_prompt:      The raw user request.
        available_tables: Table names available in the workspace.
        history:          Recent conversation history (last few turns).
        table_schemas:    Actual column names per table, e.g.
                          {"Deals": ["Stage", "Amount", "Lead Source", …], …}.
                          When provided, these are injected into the SQL-generation
                          prompt so the LLM uses real column names and never
                          hallucinates non-existent ones.
        """
        tables_str = "\n".join(f'  - "{t}"' for t in available_tables)

        # Build the schema block — the most important data-accuracy guard
        if table_schemas:
            schema_lines = []
            for tname, cols in table_schemas.items():
                col_str = " | ".join(f'"{c}"' for c in cols)
                schema_lines.append(f'  "{tname}": {col_str}')
            schema_block = (
                "\nACTUAL COLUMN NAMES PER TABLE "
                "(use ONLY these — do NOT invent column names):\n"
                + "\n".join(schema_lines)
            )
        else:
            schema_block = ""

        system = f"""You are a QUERY PLANNING AGENT for a Zoho Analytics dashboard system.
Your ONLY job: analyse the user request and produce a JSON QueryPlan containing SQL queries.

AVAILABLE TABLES IN THIS DATABASE:
{tables_str}
{schema_block}

════════════════════ OUTPUT FORMAT (strict JSON, nothing else) ════════════════════
{{
  "intent": "<dashboard|chart|kpi|table|numeric|summary>",
  "title":  "<concise dashboard or chart title>",
  "components": [
    {{
      "id":          "<unique_snake_case_id>",
      "label":       "<human-readable label shown on dashboard>",
      "output_type": "<see CHART TYPES below>",
      "sql":         "<valid SQL for Zoho Analytics — see SQL RULES below>",
      "description": "<one-line description of what this component shows>"
    }}
  ]
}}

Valid output_type values:
  kpi | numeric | gauge | dial | bullet
  bar | column | horizontalBar | stacked | stackedbar | groupedBar
  butterfly | waterfall | histogram
  pie | doughnut | donut | ring | halfpie | halfring | halfdoughnut | sunburst
  line | area | multiLine | combination | combo
  scatter | bubble | packedbubble
  funnel | heatmap | radar | spider | web | polarArea | treemap
  box | violin | table | pivot | report

══════════════════════════════ RULES ══════════════════════════════════════════════
1.  Output ONLY valid JSON — no markdown fences, no prose, no explanation.
2.  Use ONLY table names from the AVAILABLE TABLES list above.
3.  Every component MUST include: id, label, output_type, sql, description.
4.  id values must be unique snake_case strings within this plan.
5.  SQL must be SELECT-only — no INSERT, UPDATE, DELETE, DROP, or DDL.
6.  ALWAYS alias every aggregate: COUNT(*) AS "Count", SUM("Amount") AS "Total Amount".
7.  Wrap every table name in double quotes: FROM "Leads", FROM "Deals".
8.  Wrap every column name in double quotes: "Lead Status", "Stage", "Amount".
9.  Use ORDER BY <aggregate_alias> DESC for better chart sorting.
10. CHART SELECTION — choose output_type that BEST fits the data shape:
    • Single KPI number (count / total)          → kpi or numeric
    • A percentage or score out of 100           → gauge or dial
    • Pipeline stages / conversion funnel        → funnel  *** ALWAYS for stage/pipeline ***
    • Distribution of categories (3–8 items)     → pie or doughnut
    • Distribution of categories (>8 items)      → bar or horizontalBar
    • Trend over time / date series              → line or area
    • Multiple metrics over the same time axis   → multiLine or combination
    • Side-by-side comparison                    → groupedBar or butterfly
    • Part-to-whole composition across series    → stacked
    • Incremental / cumulative financial change  → waterfall
    • Correlation between two numeric columns    → scatter or bubble
    • Hierarchy / nested categories              → treemap or sunburst
    • Multi-attribute performance profile        → radar
    • Density / intensity matrix (2 categoricals)→ heatmap
    • Distribution shape of raw values           → histogram or box or violin
    • Categorical radial / polar comparison      → polarArea
    • List of individual records                 → table or report
11. DASHBOARD requests — MANDATORY MINIMUMS (scale up for broader prompts):
    a) AT LEAST 4 KPI components (output_type = kpi/numeric/gauge/dial/bullet).
       Place them FIRST in the components array.
    b) AT LEAST 5 chart/table/report components (diverse types — bar, pie, funnel,
       line, scatter, treemap, heatmap, table, etc.).
    c) Total ≥ 9 components (aim for 10–12 for a rich dashboard).
    d) Always end with 1 table or report component showing raw top records.
    e) Use a MIX of chart types — never repeat the same chart type twice.
12. Single CHART requests → exactly 1 component with the best chart type.
13. NUMERIC requests ("how many?", "what is the total?") → 1 kpi component.
14. SUMMARY requests → AT LEAST 4 kpi components + AT LEAST 2 chart components
    + 1 table of top records (minimum 7 components total).
15. For KPI: use a single aggregate — SELECT COUNT(*) AS "Total" or SUM(...) AS "Total".
16. For GROUP BY charts: include the grouped column and the aggregate — 2 columns only.
17. For multi-series charts (stacked, multiLine, groupedBar): select 1 label column + ≥2 numeric columns.
18. For table/report: SELECT * or named columns with LIMIT 50.
19. For heatmap: SELECT row_col, col_col, metric_col — exactly 3 columns.
20. For scatter/bubble: SELECT x_col, y_col [, size_col] — 2 or 3 numeric columns.
21. NEVER add LIMIT to GROUP BY / aggregation queries.
22. LIMIT 30 on raw row queries (no GROUP BY).
23. For date trends use YEAR("Date Column") or DATE_FORMAT("Date Column", '%Y-%m').
24. DO NOT use CURRENT_DATE, CURDATE(), NOW(), GETDATE(), TODAY() or similar runtime date functions.
    If user asks "this month"/"recent", use grouping by DATE_FORMAT("Created Time", '%Y-%m')
    and return latest grouped periods instead of runtime date predicates.
25. Keep SQL simple and flat — no subqueries, no CTEs, no JOINs unless essential.
26. If the user asks about revenue / amount, label it "Total Revenue" or "Pipeline Value".
27. RETURN ONLY VALID JSON — no markdown, no explanation, no backticks."""

        messages = [{"role": "system", "content": system}]

        # Include recent context (last 4 messages)
        if history:
            for h in history[-4:]:
                messages.append({"role": h["role"], "content": h["content"]})

        messages.append({"role": "user", "content": user_prompt})

        try:
            completion = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
                top_p=0.9,
            )
            raw = completion.choices[0].message.content.strip()
            plan = self._parse_json(raw)
            return self._validate_plan(plan, available_tables, user_prompt)

        except Exception as e:
            print(f"[QueryAgent] Error: {e}")
            fb = self._fallback_plan(user_prompt, available_tables)
            return self._validate_plan(fb, available_tables, user_prompt)

    # ─────────────────────────────────────────────────────────────────────────
    def _parse_json(self, text: str) -> dict:
        """Extract and parse JSON robustly, including light repair for common LLM formatting issues."""
        raw = (text or "").strip()

        # Strip markdown fences
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        # Keep the outer-most JSON object only
        start = raw.find("{")
        end = raw.rfind("}")
        candidate = raw[start:end + 1] if start != -1 and end > start else raw

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Repair common issues: smart quotes + trailing commas
        repaired = candidate
        repaired = repaired.replace("“", '"').replace("”", '"').replace("’", "'")
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            # Last attempt: remove control characters
            repaired2 = re.sub(r"[\x00-\x1F]", "", repaired)
            return json.loads(repaired2)

    # ── All recognised output_type values ────────────────────────────────────
    _VALID_OUTPUT_TYPES: frozenset[str] = frozenset({
        "kpi", "numeric", "gauge", "dial", "bullet",
        "bar", "column", "horizontalBar", "stacked", "stackedbar",
        "groupedBar", "butterfly", "waterfall", "histogram",
        "pie", "doughnut", "donut", "ring", "halfpie", "halfring",
        "halfdoughnut", "sunburst", "treemap",
        "line", "area", "multiLine", "combination", "combo",
        "scatter", "bubble", "packedbubble",
        "funnel", "heatmap", "radar", "spider", "web",
        "box", "violin", "polarArea",
        "table", "pivot", "report",
    })

    _OUTPUT_TYPE_ALIASES: dict[str, str] = {
        "barr": "bar",
        "bars": "bar",
        "vertbar": "bar",
        "verticalbar": "bar",
        "hbar": "horizontalBar",
        "horizontalbar": "horizontalBar",
        "groupbar": "groupedBar",
        "groupedbar": "groupedBar",
        "multiline": "multiLine",
        "polarea": "polarArea",
        "polar": "polarArea",
        "donught": "doughnut",
        "dougnut": "doughnut",
        "donutchart": "donut",
        "halfringchart": "halfring",
        "packedbubblechart": "packedbubble",
    }

    @classmethod
    def _normalize_output_type(cls, raw_type: str | None) -> str:
        if not raw_type:
            return "bar"
        txt = str(raw_type).strip()
        if txt in cls._VALID_OUTPUT_TYPES:
            return txt
        low = txt.lower()
        if low in cls._VALID_OUTPUT_TYPES:
            return low

        canonical = re.sub(r"[^a-z0-9]", "", low)
        if canonical in cls._OUTPUT_TYPE_ALIASES:
            return cls._OUTPUT_TYPE_ALIASES[canonical]

        # Case-insensitive exact match for camelCase known types
        for known in cls._VALID_OUTPUT_TYPES:
            if known.lower() == low:
                return known

        return "bar"

    def _validate_plan(self, plan: dict, available_tables: list[str], user_prompt: str = "") -> dict:
        """
        Validate and auto-fix common issues in a QueryPlan.
        Raises ValueError when the plan structure is fundamentally broken.
        """
        if not isinstance(plan, dict):
            raise ValueError("Plan is not a dict")
        if "components" not in plan or not plan["components"]:
            raise ValueError("No components in plan")

        for i, comp in enumerate(plan["components"]):
            # Normalize and correct unknown output_type values
            original_otype = comp.get("output_type")
            normalized_otype = self._normalize_output_type(original_otype)
            if original_otype != normalized_otype:
                print(
                    f"[QueryAgent] output_type '{original_otype}' in comp {i} "
                    f"→ normalized to '{normalized_otype}'"
                )
            comp["output_type"] = normalized_otype

            # Ensure unique id
            if not comp.get("id"):
                comp["id"] = f"comp_{i}"

            # SQL is mandatory
            if not comp.get("sql"):
                raise ValueError(f"Component {i} ('{comp.get('id')}') has no SQL")

            # Ensure label
            if not comp.get("label"):
                comp["label"] = comp["id"].replace("_", " ").title()

        # Infer intent if missing
        if not plan.get("intent"):
            plan["intent"] = "dashboard" if len(plan["components"]) > 1 else "chart"

        # Infer title if missing
        if not plan.get("title"):
            plan["title"] = "Analytics Dashboard"

        return self._postprocess_plan(plan, user_prompt)

    def _postprocess_plan(self, plan: dict, user_prompt: str) -> dict:
        """
        Final quality pass:
          1) Dashboard defaults prefer useful/core chart families.
          2) Explicit chart requests (non-dashboard) keep only requested chart types.
        """
        components = plan.get("components", [])
        if not components:
            return plan

        requested_types = self._requested_output_types(user_prompt)
        prompt_lower = (user_prompt or "").lower()

        is_dashboard_intent = plan.get("intent") == "dashboard"
        looks_full_dashboard_prompt = any(
            token in prompt_lower
            for token in ("dashboard", "full", "complete", "overview", "entire")
        )

        # Non-dashboard + explicit chart request: keep only what user asked for.
        if requested_types and (not is_dashboard_intent) and (not looks_full_dashboard_prompt):
            filtered = [c for c in components if c.get("output_type") in requested_types]
            if filtered:
                plan["components"] = filtered
            else:
                # If model missed requested type entirely, force first component to first requested type.
                components[0]["output_type"] = next(iter(requested_types))
                plan["components"] = [components[0]]
                plan["intent"] = "chart"
            return plan

        # Dashboard defaults: map exotic chart types to practical defaults unless explicitly requested.
        if is_dashboard_intent:
            practical_defaults = {
                "bar", "column", "horizontalBar",
                "line", "area",
                "pie", "doughnut", "donut", "ring",
                "funnel",
                "scatter", "bubble",
                "kpi", "numeric", "gauge", "dial", "bullet",
                "table", "pivot", "report",
            }
            remap = {
                "histogram": "bar",
                "waterfall": "bar",
                "stacked": "bar",
                "stackedbar": "bar",
                "groupedBar": "bar",
                "butterfly": "horizontalBar",
                "multiLine": "line",
                "combination": "line",
                "combo": "line",
                "packedbubble": "bubble",
                "sunburst": "pie",
                "treemap": "bar",
                "heatmap": "bar",
                "radar": "line",
                "spider": "line",
                "web": "line",
                "polarArea": "pie",
                "box": "bar",
                "violin": "bar",
                "halfpie": "pie",
                "halfring": "doughnut",
                "halfdoughnut": "doughnut",
            }

            for comp in components:
                otype = comp.get("output_type", "bar")
                if otype in practical_defaults:
                    continue
                if otype in requested_types:
                    continue
                comp["output_type"] = remap.get(otype, "bar")

        # SQL compatibility pass: strip unsupported runtime date predicates
        for comp in components:
            sql = comp.get("sql", "")
            fixed_sql = self._sanitize_sql_for_zoho(sql)
            if fixed_sql != sql:
                print(f"[QueryAgent] SQL sanitized for Zoho compatibility in '{comp.get('id', '?')}'")
                comp["sql"] = fixed_sql

        return plan

    def _sanitize_sql_for_zoho(self, sql: str) -> str:
        """Remove runtime-date predicates that commonly fail on Zoho SQL dialect."""
        if not sql:
            return sql

        s = sql
        if not re.search(r"CURRENT_DATE|CURDATE\s*\(|NOW\s*\(|GETDATE\s*\(|TODAY\s*\(", s, re.IGNORECASE):
            return s

        # Remove WHERE clause if it contains unsupported runtime date functions.
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

        s = where_pat.sub(_replace_where, s)
        return re.sub(r"\s+", " ", s).strip()

    def _requested_output_types(self, prompt: str) -> set[str]:
        """Extract explicitly requested chart families from user text."""
        p = (prompt or "").lower()
        requested: set[str] = set()

        patterns = {
            "kpi": [r"\bkpi\b", r"\bmetric\b", r"\bscore\b"],
            "line": [r"\bline\s*chart\b", r"\bline\s*graph\b", r"\btrend\b"],
            "bar": [r"\bbar\s*chart\b", r"\bbar\s*graph\b", r"\bcolumn\s*chart\b"],
            "pie": [r"\bpie\s*chart\b", r"\bdoughnut\b", r"\bdonut\b", r"\bring\s*chart\b"],
            "funnel": [r"\bfunnel\b", r"\bpipeline\b"],
            "table": [r"\btable\b", r"\breport\b", r"\bpivot\b"],
            "scatter": [r"\bscatter\b"],
            "bubble": [r"\bbubble\b"],
            "pivot": [r"\bpivot\b"],
            "report": [r"\breport\b"],
        }

        for otype, regs in patterns.items():
            if any(re.search(rg, p) for rg in regs):
                requested.add(otype)

        return requested

    def _fallback_plan(self, prompt: str, tables: list[str]) -> dict:
        """
        Keyword-based fallback plan used when Groq is unavailable or fails.
        Produces properly aliased SQL that always returns clean column names.
        """
        q          = prompt.lower()
        t_leads    = self._pick_best_table(tables, ["leads"], ["lead", "notes"])
        t_deals    = self._pick_best_table(tables, ["deals"], ["deal", "notes"])
        t_contacts = self._pick_best_table(tables, ["contacts"], ["contact", "notes"])
        t_accounts = self._pick_best_table(tables, ["accounts"], ["account", "notes"])

        if "lead" in q:
            return {
                "intent": "chart",
                "title":  "Leads Analysis",
                "components": [{
                    "id":          "leads_by_status",
                    "label":       "Leads by Status",
                    "output_type": "pie",
                    "sql":         (
                        f'SELECT "Lead Status", COUNT(*) AS "Count" '
                        f'FROM "{t_leads}" '
                        f'GROUP BY "Lead Status" '
                        f'ORDER BY "Count" DESC'
                    ),
                    "description": "Lead distribution by status",
                }],
            }

        if "deal" in q or "revenue" in q or "pipeline" in q:
            return {
                "intent": "chart",
                "title":  "Deal Pipeline",
                "components": [{
                    "id":          "deals_by_stage",
                    "label":       "Deals by Stage",
                    "output_type": "funnel",
                    "sql":         (
                        f'SELECT "Stage", COUNT(*) AS "Count" '
                        f'FROM "{t_deals}" '
                        f'GROUP BY "Stage" '
                        f'ORDER BY "Count" DESC'
                    ),
                    "description": "Deal stage funnel",
                }],
            }

        if "contact" in q:
            return {
                "intent": "chart",
                "title":  "Contacts Overview",
                "components": [{
                    "id":          "contacts_by_source",
                    "label":       "Contacts by Lead Source",
                    "output_type": "bar",
                    "sql":         (
                        f'SELECT "Lead Source", COUNT(*) AS "Count" '
                        f'FROM "{t_contacts}" '
                        f'GROUP BY "Lead Source" '
                        f'ORDER BY "Count" DESC'
                    ),
                    "description": "Contact distribution by lead source",
                }],
            }

        # Default: CRM overview dashboard — meets the ≥4 KPI + ≥5 chart minimum
        return {
            "intent": "dashboard",
            "title":  "CRM Overview",
            "components": [
                # ── 4 KPIs ───────────────────────────────────────────────────
                {
                    "id":          "total_leads",
                    "label":       "Total Leads",
                    "output_type": "kpi",
                    "sql":         f'SELECT COUNT(*) AS "Total Leads" FROM "{t_leads}"',
                    "description": "Total number of leads",
                },
                {
                    "id":          "total_deals",
                    "label":       "Total Deals",
                    "output_type": "kpi",
                    "sql":         f'SELECT COUNT(*) AS "Total Deals" FROM "{t_deals}"',
                    "description": "Total number of deals",
                },
                {
                    "id":          "total_contacts",
                    "label":       "Total Contacts",
                    "output_type": "kpi",
                    "sql":         f'SELECT COUNT(*) AS "Total Contacts" FROM "{t_contacts}"',
                    "description": "Total number of contacts",
                },
                {
                    "id":          "total_accounts",
                    "label":       "Total Accounts",
                    "output_type": "kpi",
                    "sql":         f'SELECT COUNT(*) AS "Total Accounts" FROM "{t_accounts}"',
                    "description": "Total number of accounts",
                },
                # ── 5+ Charts / Tables ───────────────────────────────────────
                {
                    "id":          "leads_by_status",
                    "label":       "Leads by Status",
                    "output_type": "pie",
                    "sql":         (
                        f'SELECT "Lead Status", COUNT(*) AS "Count" '
                        f'FROM "{t_leads}" '
                        f'GROUP BY "Lead Status" '
                        f'ORDER BY "Count" DESC'
                    ),
                    "description": "Lead status distribution",
                },
                {
                    "id":          "deals_pipeline",
                    "label":       "Deal Pipeline",
                    "output_type": "funnel",
                    "sql":         (
                        f'SELECT "Stage", COUNT(*) AS "Count" '
                        f'FROM "{t_deals}" '
                        f'GROUP BY "Stage" '
                        f'ORDER BY "Count" DESC'
                    ),
                    "description": "Pipeline deal stage funnel",
                },
                {
                    "id":          "leads_by_source",
                    "label":       "Leads by Source",
                    "output_type": "bar",
                    "sql":         (
                        f'SELECT "Lead Source", COUNT(*) AS "Count" '
                        f'FROM "{t_leads}" '
                        f'GROUP BY "Lead Source" '
                        f'ORDER BY "Count" DESC'
                    ),
                    "description": "Lead source breakdown",
                },
                {
                    "id":          "leads_trend",
                    "label":       "Leads Created Over Time",
                    "output_type": "area",
                    "sql":         (
                        f'SELECT DATE_FORMAT("Created Time", \'%Y-%m\') AS "Month", '
                        f'COUNT(*) AS "Count" '
                        f'FROM "{t_leads}" '
                        f'GROUP BY "Month" '
                        f'ORDER BY "Month" ASC'
                    ),
                    "description": "Monthly lead creation trend",
                },
                {
                    "id":          "deals_by_source",
                    "label":       "Deals by Lead Source",
                    "output_type": "horizontalBar",
                    "sql":         (
                        f'SELECT "Lead Source", COUNT(*) AS "Count" '
                        f'FROM "{t_deals}" '
                        f'GROUP BY "Lead Source" '
                        f'ORDER BY "Count" DESC'
                    ),
                    "description": "Deal distribution by lead source",
                },
                {
                    "id":          "recent_leads",
                    "label":       "Recent Leads",
                    "output_type": "table",
                    "sql":         (
                        f'SELECT "First Name", "Last Name", "Email", "Lead Status", '
                        f'"Lead Source", "Created Time" '
                        f'FROM "{t_leads}" ORDER BY "Created Time" DESC LIMIT 20'
                    ),
                    "description": "Latest 20 leads",
                },
            ],
        }

    def _pick_best_table(self, tables: list[str], preferred_exact: list[str], soft_match: list[str]) -> str:
        """Pick core CRM table names first, avoiding note/audit tables when possible."""
        if not tables:
            return preferred_exact[0].title()

        lowered = {t.lower(): t for t in tables}

        # 1) Exact preferred table names first (e.g., "Leads" over "Lead Notes")
        for name in preferred_exact:
            if name.lower() in lowered:
                return lowered[name.lower()]

        # 2) Soft match but avoid note/history/log helper tables
        for t in tables:
            lt = t.lower()
            if any(tok in lt for tok in soft_match) and not any(bad in lt for bad in ("note", "history", "log")):
                return t

        # 3) Last resort soft match
        for t in tables:
            lt = t.lower()
            if any(tok in lt for tok in soft_match):
                return t

        return tables[0]
