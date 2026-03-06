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

        # Deterministic handlers for high-priority user intents.
        # This guarantees predictable output for common asks like:
        # - "Create a dashboard of leads"
        # - "Show me a report table of all leads with name email and status"
        deterministic = self._deterministic_plan_for_prompt(
            user_prompt=user_prompt,
            available_tables=available_tables,
            table_schemas=table_schemas,
        )
        if deterministic:
            return self._validate_plan(deterministic, available_tables, user_prompt)

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
    # Deterministic planners for requirement-critical prompts
    # ─────────────────────────────────────────────────────────────────────────

    def _deterministic_plan_for_prompt(
        self,
        user_prompt: str,
        available_tables: list[str],
        table_schemas: dict[str, list[str]] | None,
    ) -> dict | None:
        prompt = (user_prompt or "").lower()

        # "Add one more chart in this dashboard" should never trigger
        # a full dashboard deterministic template.
        if self._is_incremental_chart_add_request(prompt):
            return self._build_incremental_chart_plan(available_tables, table_schemas, prompt)

        is_dashboard_req = bool(
            re.search(r"\b(dashboard|overview|complete|full|entire)\b", prompt)
        )
        is_table_req = bool(re.search(r"\b(table|report|list|rows?)\b", prompt))
        is_leads_req = "lead" in prompt
        is_kpi_req = bool(re.search(r"\b(kpi|metric|number|count|total)\b", prompt))

        # KPI-style asks should not be downgraded into table reports,
        # even when users include the word "report" in natural language.
        if is_leads_req and is_kpi_req:
            return self._build_leads_kpi_plan(available_tables, table_schemas, prompt)

        if is_table_req and is_leads_req:
            return self._build_leads_table_plan(available_tables, table_schemas, prompt)

        if is_dashboard_req and is_leads_req:
            return self._build_leads_dashboard_plan(available_tables, table_schemas)

        return None

    def _build_leads_kpi_plan(
        self,
        tables: list[str],
        table_schemas: dict[str, list[str]] | None,
        prompt: str,
    ) -> dict:
        """Deterministic single-KPI planner for leads-focused metric requests."""
        t_leads = self._pick_best_table(tables, ["leads"], ["lead"])
        cols = (table_schemas or {}).get(t_leads, [])

        converted_deal_col = self._choose_column(cols, ["converted deal", "converted_deal"])
        is_converted_col = self._choose_column(cols, ["is converted", "converted"])

        asks_converted = bool(re.search(r"\b(converted|conversion|converted\s+to\s+deals?)\b", prompt))
        wants_bullet = bool(re.search(r"\bbullet\b", prompt or "", re.IGNORECASE))

        if asks_converted and converted_deal_col:
            sql = (
                f'SELECT COUNT("{converted_deal_col}") AS "Converted Leads" '
                f'FROM "{t_leads}"'
            )
            label = "Converted Leads"
            desc = "Leads converted into deals"
        elif asks_converted and is_converted_col:
            sql = (
                f'SELECT COUNT(CASE WHEN "{is_converted_col}" = \'true\' THEN 1 END) '
                f'AS "Converted Leads" FROM "{t_leads}"'
            )
            label = "Converted Leads"
            desc = "Leads converted into deals"
        else:
            sql = f'SELECT COUNT(*) AS "Total Leads" FROM "{t_leads}"'
            label = "Total Leads"
            desc = "Total number of leads"

        return {
            "intent": "numeric",
            "title": "Leads KPI",
            "components": [
                {
                    "id": "leads_kpi",
                    "label": label,
                    "output_type": "bullet" if wants_bullet else "kpi",
                    "sql": sql,
                    "description": desc,
                }
            ],
        }

    def _build_incremental_chart_plan(
        self,
        tables: list[str],
        table_schemas: dict[str, list[str]] | None,
        prompt: str,
    ) -> dict:
        """
        Deterministic single-chart plan for "add chart" follow-up asks.
        Keeps updates incremental instead of rebuilding whole dashboards.
        """
        t_leads = self._pick_best_table(tables, ["leads"], ["lead"])
        cols = (table_schemas or {}).get(t_leads, [])

        created_col = self._choose_column(cols, ["created time", "created", "created date"])
        converted_time_col = self._choose_column(cols, ["converted date time", "lead conversion time", "converted time"])
        converted_deal_col = self._choose_column(cols, ["converted deal", "converted_deal"])
        is_converted_col = self._choose_column(cols, ["is converted", "converted"])
        status_col = self._choose_column(cols, ["lead status", "status", "stage"])
        source_col = self._choose_column(cols, ["lead source", "source"])

        wants_month = bool(re.search(r"\b(month|monthly|monthwise|by\s+month|over\s+time|trend)\b", prompt))
        asks_converted = bool(re.search(r"\b(converted|conversion|converted\s+to\s+deals?)\b", prompt or "", re.IGNORECASE))
        year_match = re.search(r"\b(20\d{2})\b", prompt)
        year_filter = f' WHERE YEAR("{created_col}") = {year_match.group(1)}' if (created_col and year_match) else ""

        if asks_converted:
            conversion_filter = None
            if converted_deal_col:
                conversion_filter = f'"{converted_deal_col}" IS NOT NULL'
            elif is_converted_col:
                conversion_filter = f'"{is_converted_col}" = \'true\''

            # Prefer a conversion trend when user explicitly asks trend/month.
            if conversion_filter and (wants_month or converted_time_col):
                trend_col = converted_time_col or created_col
                if trend_col:
                    trend_year_filter = (
                        f' AND YEAR("{trend_col}") = {year_match.group(1)}' if year_match else ""
                    )
                    return {
                        "intent": "chart",
                        "title": "Converted Leads Trend",
                        "components": [
                            {
                                "id": "converted_leads_by_month",
                                "label": "Converted Leads by Month",
                                "output_type": "line",
                                "sql": (
                                    f'SELECT DATE_FORMAT("{trend_col}", \'%Y-%m\') AS "Month", COUNT(*) AS "Count" '
                                    f'FROM "{t_leads}" '
                                    f'WHERE {conversion_filter}{trend_year_filter} '
                                    f'GROUP BY "Month" ORDER BY "Month" ASC'
                                ),
                                "description": "Monthly lead conversions to deals",
                            }
                        ],
                    }

            if conversion_filter and source_col:
                return {
                    "intent": "chart",
                    "title": "Converted Leads",
                    "components": [
                        {
                            "id": "converted_leads_by_source",
                            "label": "Converted Leads by Source",
                            "output_type": "bar",
                            "sql": (
                                f'SELECT "{source_col}", COUNT(*) AS "Count" '
                                f'FROM "{t_leads}" '
                                f'WHERE {conversion_filter} '
                                f'GROUP BY "{source_col}" '
                                f'ORDER BY "Count" DESC'
                            ),
                            "description": "Lead conversions grouped by source",
                        }
                    ],
                }

            if conversion_filter and status_col:
                return {
                    "intent": "chart",
                    "title": "Converted Leads",
                    "components": [
                        {
                            "id": "converted_leads_by_status",
                            "label": "Converted Leads by Status",
                            "output_type": "bar",
                            "sql": (
                                f'SELECT "{status_col}", COUNT(*) AS "Count" '
                                f'FROM "{t_leads}" '
                                f'WHERE {conversion_filter} '
                                f'GROUP BY "{status_col}" '
                                f'ORDER BY "Count" DESC'
                            ),
                            "description": "Converted leads grouped by status",
                        }
                    ],
                }

        if created_col and wants_month:
            return {
                "intent": "chart",
                "title": "Leads Trend",
                "components": [
                    {
                        "id": "leads_created_by_month",
                        "label": "Leads Created by Month",
                        "output_type": "line",
                        "sql": (
                            f'SELECT DATE_FORMAT("{created_col}", \'%Y-%m\') AS "Month", COUNT(*) AS "Count" '
                            f'FROM "{t_leads}"'
                            f'{year_filter}'
                            f' GROUP BY "Month" ORDER BY "Month" ASC'
                        ),
                        "description": "Monthly leads created trend",
                    }
                ],
            }

        # Generic fallback: one category chart, still single-widget.
        group_col = status_col or created_col
        if group_col == created_col:
            sql = (
                f'SELECT DATE_FORMAT("{created_col}", \'%Y-%m\') AS "Month", COUNT(*) AS "Count" '
                f'FROM "{t_leads}" GROUP BY "Month" ORDER BY "Month" ASC'
            )
            output_type = "line"
            label = "Leads Created by Month"
        else:
            sql = (
                f'SELECT "{group_col}", COUNT(*) AS "Count" '
                f'FROM "{t_leads}" GROUP BY "{group_col}" ORDER BY "Count" DESC'
            )
            output_type = "bar"
            label = "Leads by Status"

        return {
            "intent": "chart",
            "title": "Leads Chart",
            "components": [
                {
                    "id": "incremental_leads_chart",
                    "label": label,
                    "output_type": output_type,
                    "sql": sql,
                    "description": "Single chart for incremental dashboard update",
                }
            ],
        }

    def _build_leads_table_plan(
        self,
        tables: list[str],
        table_schemas: dict[str, list[str]] | None,
        prompt: str = "",
    ) -> dict:
        t_leads = self._pick_best_table(tables, ["leads"], ["lead"])
        cols = (table_schemas or {}).get(t_leads, [])

        # Prioritize explicit user-requested fields: name, email, status.
        selected = []
        name_col = self._choose_column(cols, ["full name", "name", "first name", "last name"])
        email_col = self._choose_column(cols, ["email", "email address"])
        status_col = self._choose_column(cols, ["lead status", "status", "stage"])

        for c in (name_col, email_col, status_col):
            if c and c not in selected:
                selected.append(c)

        # Add common lead context columns if present.
        source_col = self._choose_column(cols, ["lead source", "source"])
        created_col = self._choose_column(cols, ["created time", "created", "created date"])
        converted_deal_col = self._choose_column(cols, ["converted deal", "converted_deal"])
        is_converted_col = self._choose_column(cols, ["is converted", "converted"])
        for c in (source_col, created_col):
            if c and c not in selected:
                selected.append(c)

        asks_converted = bool(
            re.search(r"\b(converted|conversion|converted\s+to\s+deals?)\b", prompt or "", re.IGNORECASE)
        )

        conversion_where = ""
        if asks_converted:
            if converted_deal_col:
                conversion_where = f' WHERE "{converted_deal_col}" IS NOT NULL'
            elif is_converted_col:
                conversion_where = f' WHERE "{is_converted_col}" = \'true\''

            for c in (converted_deal_col, is_converted_col):
                if c and c not in selected:
                    selected.append(c)

        if not selected:
            if cols:
                selected = cols[:6]
            else:
                selected = ["First Name", "Last Name", "Email", "Lead Status"]

        select_sql = ", ".join(f'"{c}"' for c in selected)
        order_col = created_col or selected[0]

        return {
            "intent": "table",
            "title": "Converted Leads Report" if asks_converted else "Leads Report",
            "components": [
                {
                    "id": "leads_report_table",
                    "label": "Converted Leads Report" if asks_converted else "Leads Report",
                    "output_type": "table",
                    "sql": (
                        f"SELECT {select_sql} "
                        f"FROM \"{t_leads}\" "
                        f"{conversion_where} "
                        f"ORDER BY \"{order_col}\" DESC LIMIT 50"
                    ),
                    "description": (
                        "Lead records that converted to deals"
                        if asks_converted
                        else "Lead records with requested fields"
                    ),
                }
            ],
        }

    def _build_leads_dashboard_plan(
        self,
        tables: list[str],
        table_schemas: dict[str, list[str]] | None,
    ) -> dict:
        t_leads = self._pick_best_table(tables, ["leads"], ["lead"])
        cols = (table_schemas or {}).get(t_leads, [])

        status_col = self._choose_column(cols, ["lead status", "status", "stage"])
        source_col = self._choose_column(cols, ["lead source", "source"])
        created_col = self._choose_column(cols, ["created time", "created", "created date"])
        owner_col = self._choose_column(cols, ["lead owner", "owner"])
        city_col = self._choose_column(cols, ["city"])
        email_col = self._choose_column(cols, ["email", "email address"])
        name_col = self._choose_column(cols, ["full name", "name", "first name", "last name"])

        table_fields = []
        for c in (name_col, email_col, status_col, source_col, created_col):
            if c and c not in table_fields:
                table_fields.append(c)
        if not table_fields:
            table_fields = cols[:6] if cols else ["First Name", "Last Name", "Email", "Lead Status"]

        components = [
            {
                "id": "total_leads",
                "label": "Total Leads",
                "output_type": "kpi",
                "sql": f'SELECT COUNT(*) AS "Total Leads" FROM "{t_leads}"',
                "description": "Total number of leads",
            },
            {
                "id": "contactable_leads",
                "label": "Contactable Leads",
                "output_type": "numeric",
                "sql": (
                    f'SELECT COUNT("{email_col}") AS "Contactable Leads" FROM "{t_leads}"'
                    if email_col
                    else f'SELECT COUNT(*) AS "Contactable Leads" FROM "{t_leads}"'
                ),
                "description": "Leads with email available",
            },
            {
                "id": "lead_status_count",
                "label": "Lead Statuses",
                "output_type": "gauge",
                "sql": (
                    f'SELECT COUNT(DISTINCT "{status_col}") AS "Lead Statuses" FROM "{t_leads}"'
                    if status_col
                    else f'SELECT COUNT(*) AS "Lead Statuses" FROM "{t_leads}"'
                ),
                "description": "Number of distinct lead statuses",
            },
            {
                "id": "lead_source_count",
                "label": "Lead Sources",
                "output_type": "dial",
                "sql": (
                    f'SELECT COUNT(DISTINCT "{source_col}") AS "Lead Sources" FROM "{t_leads}"'
                    if source_col
                    else f'SELECT COUNT(*) AS "Lead Sources" FROM "{t_leads}"'
                ),
                "description": "Number of distinct lead sources",
            },
            {
                "id": "leads_by_status",
                "label": "Leads by Status",
                "output_type": "pie",
                "sql": (
                    f'SELECT "{status_col}", COUNT(*) AS "Count" '
                    f'FROM "{t_leads}" '
                    f'GROUP BY "{status_col}" '
                    f'ORDER BY "Count" DESC'
                    if status_col
                    else (
                        f'SELECT "{source_col}", COUNT(*) AS "Count" '
                        f'FROM "{t_leads}" '
                        f'GROUP BY "{source_col}" '
                        f'ORDER BY "Count" DESC'
                        if source_col
                        else f'SELECT COUNT(*) AS "Count" FROM "{t_leads}"'
                    )
                ),
                "description": "Lead distribution by status",
            },
            {
                "id": "leads_by_source",
                "label": "Leads by Source",
                "output_type": "bar",
                "sql": (
                    f'SELECT "{source_col}", COUNT(*) AS "Count" '
                    f'FROM "{t_leads}" '
                    f'GROUP BY "{source_col}" '
                    f'ORDER BY "Count" DESC'
                    if source_col
                    else (
                        f'SELECT "{status_col}", COUNT(*) AS "Count" '
                        f'FROM "{t_leads}" '
                        f'GROUP BY "{status_col}" '
                        f'ORDER BY "Count" DESC'
                        if status_col
                        else f'SELECT COUNT(*) AS "Count" FROM "{t_leads}"'
                    )
                ),
                "description": "Lead source breakdown",
            },
            {
                "id": "leads_by_owner",
                "label": "Leads by Owner",
                "output_type": "horizontalBar",
                "sql": (
                    f'SELECT "{owner_col}", COUNT(*) AS "Count" '
                    f'FROM "{t_leads}" '
                    f'GROUP BY "{owner_col}" '
                    f'ORDER BY "Count" DESC'
                    if owner_col
                    else (
                        f'SELECT "{source_col}", COUNT(*) AS "Count" '
                        f'FROM "{t_leads}" '
                        f'GROUP BY "{source_col}" '
                        f'ORDER BY "Count" DESC'
                        if source_col
                        else f'SELECT COUNT(*) AS "Count" FROM "{t_leads}"'
                    )
                ),
                "description": "Lead ownership distribution",
            },
            {
                "id": "leads_over_time",
                "label": "Leads Created Over Time",
                "output_type": "line",
                "sql": (
                    f'SELECT DATE_FORMAT("{created_col}", \'%Y-%m\') AS "Month", COUNT(*) AS "Count" '
                    f'FROM "{t_leads}" '
                    f'GROUP BY "Month" '
                    f'ORDER BY "Month" ASC'
                    if created_col
                    else (
                        f'SELECT "{status_col}", COUNT(*) AS "Count" '
                        f'FROM "{t_leads}" '
                        f'GROUP BY "{status_col}" '
                        f'ORDER BY "Count" DESC'
                        if status_col
                        else f'SELECT COUNT(*) AS "Count" FROM "{t_leads}"'
                    )
                ),
                "description": "Lead creation trend",
            },
            {
                "id": "recent_leads",
                "label": "Recent Leads",
                "output_type": "table",
                "sql": (
                    f'SELECT {", ".join(f"\"{c}\"" for c in table_fields)} '
                    f'FROM "{t_leads}" '
                    f'ORDER BY "{created_col or table_fields[0]}" DESC LIMIT 30'
                ),
                "description": "Recent lead records",
            },
        ]

        # Optional extra component for richer dashboard diversity.
        if city_col:
            components.insert(8, {
                "id": "leads_by_city",
                "label": "Leads by City",
                "output_type": "area",
                "sql": (
                    f'SELECT "{city_col}", COUNT(*) AS "Count" '
                    f'FROM "{t_leads}" '
                    f'GROUP BY "{city_col}" '
                    f'ORDER BY "Count" DESC'
                ),
                "description": "Geographic distribution of leads",
            })

        return {
            "intent": "dashboard",
            "title": "Leads Dashboard",
            "components": components,
        }

    def _choose_column(self, columns: list[str], candidates: list[str]) -> str | None:
        if not columns:
            return None

        # Exact normalized match first
        normalized = {self._norm_col(c): c for c in columns}
        for cand in candidates:
            key = self._norm_col(cand)
            if key in normalized:
                return normalized[key]

        # Soft contains match next
        for cand in candidates:
            needle = self._norm_col(cand)
            for col in columns:
                if needle and needle in self._norm_col(col):
                    return col

        return None

    @staticmethod
    def _norm_col(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", (name or "").lower())

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

        return self._postprocess_plan(plan, user_prompt, available_tables)

    def _postprocess_plan(self, plan: dict, user_prompt: str, available_tables: list[str]) -> dict:
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
        is_incremental_add = self._is_incremental_chart_add_request(prompt_lower)

        # Force follow-up add requests to a single non-KPI component.
        if is_incremental_add:
            plan["intent"] = "chart"
            non_kpi = [
                c for c in components
                if c.get("output_type") not in {"kpi", "numeric", "gauge", "dial", "bullet"}
            ]
            chosen = non_kpi[0] if non_kpi else components[0]

            if requested_types and chosen.get("output_type") not in requested_types:
                chosen["output_type"] = next(iter(requested_types))

            plan["components"] = [chosen]
            return plan

        is_dashboard_prompt = self._is_dashboard_request(prompt_lower)
        is_single_chart_prompt = self._is_single_chart_request(prompt_lower)

        # Force chart intent for explicit single-chart asks.
        if is_single_chart_prompt and not is_dashboard_prompt:
            plan["intent"] = "chart"
            if requested_types:
                filtered = [c for c in components if c.get("output_type") in requested_types]
                if filtered:
                    plan["components"] = [filtered[0]]
                else:
                    components[0]["output_type"] = next(iter(requested_types))
                    plan["components"] = [components[0]]
            else:
                plan["components"] = [components[0]]
            return plan

        is_dashboard_intent = plan.get("intent") == "dashboard"
        looks_full_dashboard_prompt = any(
            token in prompt_lower
            for token in ("dashboard", "full", "complete", "overview", "entire")
        )

        # Global rule: dashboard prompts must always return a rich dashboard.
        if is_dashboard_prompt:
            plan["intent"] = "dashboard"
            if not self._is_rich_dashboard(plan.get("components", [])):
                print("[QueryAgent] Dashboard prompt under-filled — enforcing rich fallback dashboard.")
                fallback = self._fallback_plan("dashboard overview", available_tables)
                plan["components"] = fallback.get("components", [])
                if not plan.get("title") or plan.get("title") == "Analytics Dashboard":
                    plan["title"] = fallback.get("title", "Analytics Dashboard")
            components = plan.get("components", [])
            is_dashboard_intent = True
            looks_full_dashboard_prompt = True

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

    def _is_dashboard_request(self, prompt: str) -> bool:
        if not prompt:
            return False
        if self._is_incremental_chart_add_request(prompt):
            return False
        return bool(re.search(r"\b(dashboard|overview|complete|full|entire|all metrics|summary dashboard)\b", prompt))

    def _is_incremental_chart_add_request(self, prompt: str) -> bool:
        if not prompt:
            return False

        has_widget_term = bool(re.search(r"\b(chart|graph|plot|kpi|widget|table|report)\b", prompt))
        asks_single = bool(re.search(r"\b(one|single|one\s+more|another|more)\b", prompt))
        add_verb = bool(re.search(r"\b(add|append|include|insert)\b", prompt))
        create_verb = bool(re.search(r"\b(create|make|put)\b", prompt))
        refers_existing_dash = bool(re.search(r"\b(this|same|current|existing)\s+dashboard\b", prompt))

        if not has_widget_term:
            return False

        # Strong signal: explicit add-like verb + either existing dashboard context
        # or single-item language ("add one more chart").
        if add_verb and (refers_existing_dash or asks_single):
            return True

        # For create/make/put, require explicit existing dashboard context to avoid
        # hijacking "create dashboard ..." as an incremental add request.
        if create_verb and refers_existing_dash and asks_single:
            return True

        return False

    def _is_single_chart_request(self, prompt: str) -> bool:
        if not prompt:
            return False

        explicit_single = bool(
            re.search(r"\b(only\s+one|single|just\s+one|one)\s+(chart|graph|plot)\b", prompt)
        )
        if explicit_single:
            return True

        has_chart_word = bool(re.search(r"\b(chart|graph|plot|pie\s*chart|bar\s*chart|line\s*chart|funnel)\b", prompt))
        has_multi_or_dash = bool(re.search(r"\b(charts|graphs|dashboard|overview|kpis?|table|summary|all)\b", prompt))
        return has_chart_word and not has_multi_or_dash

    def _is_rich_dashboard(self, components: list[dict]) -> bool:
        if not components or len(components) < 7:
            return False

        kpi_types = {"kpi", "numeric", "gauge", "dial", "bullet"}
        kpi_count = sum(1 for c in components if c.get("output_type") in kpi_types)
        has_table = any(c.get("output_type") in {"table", "pivot", "report"} for c in components)
        unique_types = {c.get("output_type") for c in components if c.get("output_type")}

        return kpi_count >= 3 and has_table and len(unique_types) >= 4

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
            "bullet": [r"\bbullet\b", r"\bbullet\s*kpi\b"],
            "gauge": [r"\bgauge\b"],
            "dial": [r"\bdial\b"],
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

        # Dashboard prompts should never collapse into a single chart.
        if self._is_dashboard_request(q):
            return {
                "intent": "dashboard",
                "title":  "CRM Overview",
                "components": [
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
                        "output_type": "numeric",
                        "sql":         f'SELECT COUNT(*) AS "Total Deals" FROM "{t_deals}"',
                        "description": "Total number of deals",
                    },
                    {
                        "id":          "total_contacts",
                        "label":       "Total Contacts",
                        "output_type": "gauge",
                        "sql":         f'SELECT COUNT(*) AS "Total Contacts" FROM "{t_contacts}"',
                        "description": "Total number of contacts",
                    },
                    {
                        "id":          "total_accounts",
                        "label":       "Total Accounts",
                        "output_type": "dial",
                        "sql":         f'SELECT COUNT(*) AS "Total Accounts" FROM "{t_accounts}"',
                        "description": "Total number of accounts",
                    },
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
                    "output_type": "numeric",
                    "sql":         f'SELECT COUNT(*) AS "Total Deals" FROM "{t_deals}"',
                    "description": "Total number of deals",
                },
                {
                    "id":          "total_contacts",
                    "label":       "Total Contacts",
                    "output_type": "gauge",
                    "sql":         f'SELECT COUNT(*) AS "Total Contacts" FROM "{t_contacts}"',
                    "description": "Total number of contacts",
                },
                {
                    "id":          "total_accounts",
                    "label":       "Total Accounts",
                    "output_type": "dial",
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
