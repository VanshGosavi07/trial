"""
Agent 3: Visualization Agent
──────────────────────────────
Takes a list of QueryResult dicts (from MCPDataAgent, sanitized by ValidationAgent)
and transforms each into a frontend-ready visualization spec.

Standard chart / KPI types use deterministic Python logic — fast, zero extra API
calls.  Complex "summary" responses optionally invoke Groq for a prose summary.

WidgetSpec output (per component):
{
  "chart_type": "kpi | bar | line | pie | funnel | table | …",
  "title":      "Widget title",
  "data":       { … Plotly-compatible data … },
  "w":          int   (GridStack column span),
  "h":          int   (GridStack row span)
}
"""

import re
import json
import os
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# ─── Grid layout hints ────────────────────────────────────────────────────────
_LAYOUT = {
    # ── Numeric KPIs ─────────────────────────────────────────────────────────
    "kpi":           {"w": 3, "h": 2},
    "numeric":       {"w": 3, "h": 2},
    "gauge":         {"w": 3, "h": 3},
    "dial":          {"w": 3, "h": 3},
    "bullet":        {"w": 4, "h": 2},
    # ── Pie family ───────────────────────────────────────────────────────────
    "pie":           {"w": 6, "h": 5},
    "doughnut":      {"w": 6, "h": 5},
    "donut":         {"w": 6, "h": 5},
    "ring":          {"w": 6, "h": 5},
    "halfpie":       {"w": 6, "h": 4},
    "halfring":      {"w": 6, "h": 4},
    "halfdoughnut":  {"w": 6, "h": 4},
    "sunburst":      {"w": 6, "h": 5},
    # ── Bar family ───────────────────────────────────────────────────────────
    "bar":           {"w": 6, "h": 5},
    "column":        {"w": 6, "h": 5},
    "horizontalBar": {"w": 8, "h": 5},
    "stacked":       {"w": 8, "h": 5},
    "stackedbar":    {"w": 8, "h": 5},
    "groupedBar":    {"w": 8, "h": 5},
    "butterfly":     {"w": 8, "h": 5},
    "waterfall":     {"w": 8, "h": 5},
    "histogram":     {"w": 7, "h": 5},
    # ── Line / Area / Time-series ─────────────────────────────────────────────
    "line":          {"w": 7, "h": 5},
    "area":          {"w": 7, "h": 5},
    "multiLine":     {"w": 8, "h": 5},
    "combination":   {"w": 8, "h": 5},
    "combo":         {"w": 8, "h": 5},
    # ── Scatter / Bubble ─────────────────────────────────────────────────────
    "scatter":       {"w": 7, "h": 5},
    "bubble":        {"w": 7, "h": 5},
    "packedbubble":  {"w": 7, "h": 5},
    # ── Specialized ──────────────────────────────────────────────────────────
    "funnel":        {"w": 5, "h": 6},
    "heatmap":       {"w": 8, "h": 5},
    "radar":         {"w": 6, "h": 5},
    "spider":        {"w": 6, "h": 5},
    "web":           {"w": 6, "h": 5},
    "polarArea":     {"w": 6, "h": 5},
    "treemap":       {"w": 8, "h": 5},
    "box":           {"w": 8, "h": 5},
    "violin":        {"w": 8, "h": 5},
    # ── Tables ───────────────────────────────────────────────────────────────
    "table":         {"w": 12, "h": 6},
    "pivot":         {"w": 12, "h": 6},
    "report":        {"w": 12, "h": 6},
}

_VALID_TYPES = set(_LAYOUT.keys())
_OUTPUT_TYPE_ALIASES = {
    "barr": "bar",
    "bars": "bar",
    "verticalbar": "bar",
    "hbar": "horizontalBar",
    "horizontalbar": "horizontalBar",
    "groupbar": "groupedBar",
    "groupedbar": "groupedBar",
    "multiline": "multiLine",
    "polar": "polarArea",
    "polarea": "polarArea",
    "donught": "doughnut",
    "dougnut": "doughnut",
}


class VizAgent:
    """
    Visualization Agent — pure-Python transformation, no extra LLM calls needed
    for standard charts.
    """

    def __init__(self):
        self._groq = Groq(api_key=GROQ_API_KEY)

    # ─────────────────────────────────────────────────────────────────────────
    def build_widgets(self, query_results: list[dict],
                      user_prompt: str = "") -> list[dict]:
        """
        Transform each QueryResult into a WidgetSpec.
        Returns list of WidgetSpec dicts.
        """
        widgets = []
        for qr in query_results:
            try:
                w = self._build_one(qr)
                if w:
                    widgets.append(w)
            except Exception as e:
                print(f"[VizAgent] Error building widget {qr.get('id')}: {e}")
                widgets.append(self._error_widget(qr.get("label", "Widget"), str(e)))
        return widgets

    # ─────────────────────────────────────────────────────────────────────────
    def _build_one(self, qr: dict) -> dict | None:
        otype   = self._normalize_output_type(qr.get("output_type", "bar"))
        label   = qr.get("label", "Chart")
        rows    = qr.get("rows", [])
        cols    = qr.get("columns", [])
        error   = qr.get("error")

        if error:
            return self._error_widget(label, error)

        if not rows:
            return self._empty_widget(label, otype)

        if not self._is_data_sufficient(otype, rows, cols):
            print(f"[VizAgent] Insufficient data for '{otype}' in '{label}' — using table fallback")
            return self._table_widget(label, rows, cols, {"w": 12, "h": 6})

        layout = _LAYOUT.get(otype, {"w": 6, "h": 5})

        # ── KPI family (card / gauge / dial / bullet) ─────────────────────
        if otype in ("kpi", "numeric", "gauge", "dial", "bullet"):
            return self._kpi_widget(label, rows, cols, otype, layout)

        # ── Table / Pivot ──────────────────────────────────────────────────
        if otype in ("table", "pivot"):
            return self._table_widget(label, rows, cols, layout)

        # ── Pie family ─────────────────────────────────────────────────────
        if otype in ("pie", "doughnut", "donut", "ring", "halfpie",
                     "halfring", "halfdoughnut"):
            return self._pie_widget(label, rows, cols, otype, layout)

        # ── Sunburst ───────────────────────────────────────────────────────
        if otype == "sunburst":
            return self._sunburst_widget(label, rows, cols, layout)

        # ── Funnel ─────────────────────────────────────────────────────────
        if otype == "funnel":
            return self._funnel_widget(label, rows, cols, layout)

        # ── Line / Area ────────────────────────────────────────────────────
        if otype in ("line", "area"):
            return self._line_widget(label, rows, cols, otype, layout)

        # ── Multi-Line ─────────────────────────────────────────────────────
        if otype == "multiLine":
            return self._multiline_widget(label, rows, cols, layout)

        # ── Combination (bar+line) ─────────────────────────────────────────
        if otype in ("combination", "combo"):
            return self._combination_widget(label, rows, cols, layout)

        # ── Horizontal Bar ─────────────────────────────────────────────────
        if otype == "horizontalBar":
            return self._hbar_widget(label, rows, cols, layout)

        # ── Stacked / Grouped Bar ──────────────────────────────────────────
        if otype in ("stacked", "stackedbar", "groupedBar"):
            return self._multi_bar_widget(label, rows, cols, otype, layout)

        # ── Butterfly ──────────────────────────────────────────────────────
        if otype == "butterfly":
            return self._butterfly_widget(label, rows, cols, layout)

        # ── Waterfall ──────────────────────────────────────────────────────
        if otype == "waterfall":
            return self._waterfall_widget(label, rows, cols, layout)

        # ── Treemap ────────────────────────────────────────────────────────
        if otype == "treemap":
            return self._treemap_widget(label, rows, cols, layout)

        # ── Scatter ────────────────────────────────────────────────────────
        if otype == "scatter":
            return self._scatter_widget(label, rows, cols, layout)

        # ── Bubble ─────────────────────────────────────────────────────────
        if otype == "bubble":
            return self._bubble_widget(label, rows, cols, layout)

        # ── Packed Bubble ──────────────────────────────────────────────────
        if otype == "packedbubble":
            return self._packedbubble_widget(label, rows, cols, layout)

        # ── Box Plot ───────────────────────────────────────────────────────
        if otype == "box":
            return self._box_widget(label, rows, cols, layout)

        # ── Violin ─────────────────────────────────────────────────────────
        if otype == "violin":
            return self._violin_widget(label, rows, cols, layout)

        # ── Histogram ──────────────────────────────────────────────────────
        if otype == "histogram":
            return self._histogram_widget(label, rows, cols, layout)

        # ── Polar Area ─────────────────────────────────────────────────────
        if otype == "polarArea":
            return self._polar_widget(label, rows, cols, layout)

        # ── Heatmap ────────────────────────────────────────────────────────
        if otype == "heatmap":
            return self._heatmap_widget(label, rows, cols, layout)

        # ── Radar / Spider / Web ───────────────────────────────────────────
        if otype in ("radar", "spider", "web"):
            return self._radar_widget(label, rows, cols, layout)

        # ── Report / Table ─────────────────────────────────────────────────
        if otype == "report":
            return self._table_widget(label, rows, cols, {"w": 12, "h": 6})

        # ── Default fallback: bar ──────────────────────────────────────────
        return self._bar_widget(label, rows, cols, layout)

    @staticmethod
    def _normalize_output_type(raw_type: str | None) -> str:
        if not raw_type:
            return "bar"

        txt = str(raw_type).strip()
        if txt in _VALID_TYPES:
            return txt

        low = txt.lower()
        if low in _VALID_TYPES:
            return low

        canonical = re.sub(r"[^a-z0-9]", "", low)
        if canonical in _OUTPUT_TYPE_ALIASES:
            return _OUTPUT_TYPE_ALIASES[canonical]

        return "bar"

    # ── KPI ───────────────────────────────────────────────────────────────────
    def _kpi_widget(self, title, rows, cols, otype, layout):
        if not rows or not cols:
            return self._empty_widget(title, "kpi")
        row = rows[0]
        col = cols[0]
        raw = row.get(col, "—")

        # Coerce and format the value
        if self._looks_numeric(str(raw)):
            num = self._to_num(raw)
            val = self._format_kpi_number(num)
        else:
            val = str(raw) if raw is not None else "—"

        subtitle = col if col.lower() not in title.lower() else ""

        # Gauge/Dial/Bullet need numeric payload for Plotly indicator rendering.
        if otype in ("gauge", "dial", "bullet"):
            num_val = self._to_num(raw)
            max_val = self._suggest_kpi_max(num_val)
            payload = {
                "value": num_val,
                "subtitle": subtitle,
                "icon": self._kpi_icon(title),
                "min": 0,
                "max": max_val,
            }
            if otype == "bullet":
                payload["target"] = max(1, int(max_val * 0.75))

            return {
                "chart_type": otype,
                "title": title,
                "data": payload,
                **layout,
            }

        return {
            "chart_type": otype if otype in ("kpi", "numeric") else "kpi",
            "title": title,
            "data": {
                "value":    val,
                "subtitle": subtitle,
                "icon":     self._kpi_icon(title),
            },
            **layout,
        }

    @staticmethod
    def _suggest_kpi_max(value: float | int) -> int:
        """Pick a readable gauge max slightly above the current value."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 100

        if v <= 0:
            return 100
        if v <= 10:
            return 10
        if v <= 100:
            return 100
        if v <= 1_000:
            return 1_000
        if v <= 10_000:
            return 10_000
        return int(v * 1.25)

    # ── Table ─────────────────────────────────────────────────────────────────
    def _table_widget(self, title, rows, cols, layout):
        return {
            "chart_type": "table",
            "title": title,
            "data": {"columns": cols, "rows": rows},
            **layout
        }

    # ── Pie ───────────────────────────────────────────────────────────────────
    def _pie_widget(self, title, rows, cols, otype, layout):
        label_col, val_col = self._detect_label_value(cols, rows)
        labels = [str(r.get(label_col, "?")) for r in rows]
        values = [self._to_num(r.get(val_col, 0)) for r in rows]
        return {
            "chart_type": otype,
            "title": title,
            "data": {"labels": labels, "values": values},
            **layout
        }

    # ── Funnel ────────────────────────────────────────────────────────────────
    def _funnel_widget(self, title, rows, cols, layout):
        label_col, val_col = self._detect_label_value(cols, rows)
        labels = [str(r.get(label_col, "?")) for r in rows]
        values = [self._to_num(r.get(val_col, 0)) for r in rows]
        return {
            "chart_type": "funnel",
            "title": title,
            "data": {"labels": labels, "values": values},
            **layout
        }

    # ── Line / Area ───────────────────────────────────────────────────────────
    def _line_widget(self, title, rows, cols, otype, layout):
        label_col, val_col = self._detect_label_value(cols, rows)
        labels = [str(r.get(label_col, "?")) for r in rows]
        values = [self._to_num(r.get(val_col, 0)) for r in rows]
        return {
            "chart_type": otype,
            "title": title,
            "data": {"labels": labels, "values": values},
            **layout
        }

    # ── Bar (vertical) ────────────────────────────────────────────────────────
    def _bar_widget(self, title, rows, cols, layout):
        label_col, val_col = self._detect_label_value(cols, rows)
        labels = [str(r.get(label_col, "?")) for r in rows]
        values = [self._to_num(r.get(val_col, 0)) for r in rows]
        return {
            "chart_type": "bar",
            "title": title,
            "data": {"labels": labels, "values": values},
            **layout
        }

    # ── Horizontal Bar ────────────────────────────────────────────────────────
    def _hbar_widget(self, title, rows, cols, layout):
        label_col, val_col = self._detect_label_value(cols, rows)
        labels = [str(r.get(label_col, "?")) for r in rows]
        values = [self._to_num(r.get(val_col, 0)) for r in rows]
        return {
            "chart_type": "horizontalBar",
            "title": title,
            "data": {"labels": labels, "values": values},
            **layout
        }

    # ── Stacked / Grouped Bar ─────────────────────────────────────────────────
    def _multi_bar_widget(self, title, rows, cols, otype, layout):
        # First col = category, remaining numeric = series
        if len(cols) < 2:
            return self._bar_widget(title, rows, cols, layout)

        label_col = cols[0]
        num_cols  = [c for c in cols[1:] if self._col_is_numeric(c, rows)]

        if not num_cols:
            return self._bar_widget(title, rows, cols, layout)

        labels  = [str(r.get(label_col, "?")) for r in rows]
        series  = [{"name": nc, "values": [self._to_num(r.get(nc, 0)) for r in rows]}
                   for nc in num_cols]
        return {
            "chart_type": otype,
            "title": title,
            "data": {"labels": labels, "series": series},
            **layout
        }

    # ── MultiLine ─────────────────────────────────────────────────────────────
    def _multiline_widget(self, title, rows, cols, layout):
        """Multiple line series — first col = x labels, remaining numeric = series."""
        if len(cols) < 2:
            return self._line_widget(title, rows, cols, "line", layout)
        label_col = cols[0]
        num_cols  = [c for c in cols[1:] if self._col_is_numeric(c, rows)]
        if not num_cols:
            return self._line_widget(title, rows, cols, "line", layout)
        labels = [str(r.get(label_col, "?")) for r in rows]
        series = [{"name": nc, "values": [self._to_num(r.get(nc, 0)) for r in rows]}
                  for nc in num_cols]
        return {
            "chart_type": "multiLine",
            "title": title,
            "data": {"labels": labels, "series": series},
            **layout
        }

    # ── Combination (bar + line) ──────────────────────────────────────────────
    def _combination_widget(self, title, rows, cols, layout):
        """First N-1 cols → bars, last numeric col → line (secondary axis)."""
        if len(cols) < 2:
            return self._bar_widget(title, rows, cols, layout)
        label_col = cols[0]
        num_cols  = [c for c in cols[1:] if self._col_is_numeric(c, rows)]
        if len(num_cols) < 2:
            return self._multi_bar_widget(title, rows, cols, "groupedBar", layout)
        labels = [str(r.get(label_col, "?")) for r in rows]
        series = []
        for i, nc in enumerate(num_cols):
            is_line = (i == len(num_cols) - 1)
            series.append({
                "name":    nc,
                "values": [self._to_num(r.get(nc, 0)) for r in rows],
                "type":   "line" if is_line else "bar"
            })
        return {
            "chart_type": "combination",
            "title": title,
            "data": {"labels": labels, "series": series},
            **layout
        }

    # ── Bubble ────────────────────────────────────────────────────────────────
    def _bubble_widget(self, title, rows, cols, layout):
        if len(cols) < 2:
            return self._bar_widget(title, rows, cols, layout)
        xc = cols[0]
        yc = cols[1] if len(cols) > 1 else cols[0]
        sc = cols[2] if len(cols) > 2 else yc
        lc = None
        # Find a text label column (non-numeric)
        for c in cols:
            if not self._col_is_numeric(c, rows):
                lc = c; break
        return {
            "chart_type": "bubble",
            "title": title,
            "data": {
                "x":      [self._to_num(r.get(xc, 0)) for r in rows],
                "y":      [self._to_num(r.get(yc, 0)) for r in rows],
                "sizes":  [self._to_num(r.get(sc, 0)) for r in rows],
                "labels": [str(r.get(lc, "")) for r in rows] if lc else []
            },
            **layout
        }

    # ── Packed Bubble ─────────────────────────────────────────────────────────
    def _packedbubble_widget(self, title, rows, cols, layout):
        label_col, val_col = self._detect_label_value(cols, rows)
        labels = [str(r.get(label_col, "?")) for r in rows]
        values = [self._to_num(r.get(val_col, 0)) for r in rows]
        return {
            "chart_type": "packedbubble",
            "title": title,
            "data": {"labels": labels, "values": values},
            **layout
        }

    # ── Butterfly (back-to-back horizontal bar) ───────────────────────────────
    def _butterfly_widget(self, title, rows, cols, layout):
        if len(cols) < 3:
            return self._hbar_widget(title, rows, cols, layout)
        label_col = cols[0]
        num_cols  = [c for c in cols[1:] if self._col_is_numeric(c, rows)]
        if len(num_cols) < 2:
            return self._hbar_widget(title, rows, cols, layout)
        labels = [str(r.get(label_col, "?")) for r in rows]
        return {
            "chart_type": "butterfly",
            "title": title,
            "data": {
                "labels": labels,
                "series": [
                    {"name": num_cols[0], "values": [self._to_num(r.get(num_cols[0], 0)) for r in rows]},
                    {"name": num_cols[1], "values": [self._to_num(r.get(num_cols[1], 0)) for r in rows]},
                ]
            },
            **layout
        }

    # ── Waterfall ─────────────────────────────────────────────────────────────
    def _waterfall_widget(self, title, rows, cols, layout):
        label_col, val_col = self._detect_label_value(cols, rows)
        labels = [str(r.get(label_col, "?")) for r in rows]
        values = [self._to_num(r.get(val_col, 0)) for r in rows]
        measure = ["relative"] * len(values)
        if measure:
            measure[-1] = "total"
        return {
            "chart_type": "waterfall",
            "title": title,
            "data": {"labels": labels, "values": values, "measure": measure},
            **layout
        }

    # ── Treemap ───────────────────────────────────────────────────────────────
    def _treemap_widget(self, title, rows, cols, layout):
        if len(cols) >= 3:
            # col0=parent, col1=child/label, col2=value
            parent_col, label_col, val_col = cols[0], cols[1], cols[2]
            labels  = [str(r.get(label_col, "?")) for r in rows]
            values  = [self._to_num(r.get(val_col, 0)) for r in rows]
            parents = [str(r.get(parent_col, "")) for r in rows]
            return {
                "chart_type": "treemap",
                "title": title,
                "data": {"labels": labels, "values": values, "parents": parents},
                **layout
            }
        label_col, val_col = self._detect_label_value(cols, rows)
        labels = [str(r.get(label_col, "?")) for r in rows]
        values = [self._to_num(r.get(val_col, 0)) for r in rows]
        return {
            "chart_type": "treemap",
            "title": title,
            "data": {"labels": labels, "values": values, "parents": [""] * len(labels)},
            **layout
        }

    # ── Sunburst ──────────────────────────────────────────────────────────────
    def _sunburst_widget(self, title, rows, cols, layout):
        if len(cols) >= 3:
            parent_col, label_col, val_col = cols[0], cols[1], cols[2]
            labels  = [str(r.get(label_col, "?")) for r in rows]
            values  = [self._to_num(r.get(val_col, 0)) for r in rows]
            parents = [str(r.get(parent_col, "")) for r in rows]
            return {
                "chart_type": "sunburst",
                "title": title,
                "data": {"labels": labels, "values": values, "parents": parents},
                **layout
            }
        label_col, val_col = self._detect_label_value(cols, rows)
        labels = [str(r.get(label_col, "?")) for r in rows]
        values = [self._to_num(r.get(val_col, 0)) for r in rows]
        return {
            "chart_type": "sunburst",
            "title": title,
            "data": {"labels": labels, "values": values, "parents": [""] * len(labels)},
            **layout
        }

    # ── Box Plot ──────────────────────────────────────────────────────────────
    def _box_widget(self, title, rows, cols, layout):
        num_cols  = [c for c in cols if self._col_is_numeric(c, rows)]
        cat_col   = next((c for c in cols if not self._col_is_numeric(c, rows)), None)
        if cat_col and num_cols:
            # Grouped box: one series per category value
            cats   = sorted(set(str(r.get(cat_col, "")) for r in rows))
            series = []
            for cat in cats:
                cat_rows = [r for r in rows if str(r.get(cat_col, "")) == cat]
                series.append({"name": cat,
                               "values": [self._to_num(r.get(num_cols[0], 0)) for r in cat_rows]})
            return {
                "chart_type": "box",
                "title": title,
                "data": {"series": series},
                **layout
            }
        # One box per numeric column
        series = [{"name": nc, "values": [self._to_num(r.get(nc, 0)) for r in rows]}
                  for nc in (num_cols or cols[:1])]
        return {
            "chart_type": "box",
            "title": title,
            "data": {"series": series},
            **layout
        }

    # ── Violin ────────────────────────────────────────────────────────────────
    def _violin_widget(self, title, rows, cols, layout):
        """Reuse same data structure as box, frontend renders as violin."""
        w = self._box_widget(title, rows, cols, layout)
        w["chart_type"] = "violin"
        return w

    # ── Histogram ─────────────────────────────────────────────────────────────
    def _histogram_widget(self, title, rows, cols, layout):
        _, val_col = self._detect_label_value(cols, rows)
        values = [self._to_num(r.get(val_col, 0)) for r in rows]
        return {
            "chart_type": "histogram",
            "title": title,
            "data": {"values": values},
            **layout
        }

    # ── Polar Area ────────────────────────────────────────────────────────────
    def _polar_widget(self, title, rows, cols, layout):
        label_col, val_col = self._detect_label_value(cols, rows)
        labels = [str(r.get(label_col, "?")) for r in rows]
        values = [self._to_num(r.get(val_col, 0)) for r in rows]
        return {
            "chart_type": "polarArea",
            "title": title,
            "data": {"labels": labels, "values": values},
            **layout
        }

    # ── Scatter ───────────────────────────────────────────────────────────────
    def _scatter_widget(self, title, rows, cols, layout):
        if len(cols) < 2:
            return self._bar_widget(title, rows, cols, layout)
        xc, yc = cols[0], cols[1]
        lc     = cols[2] if len(cols) > 2 else None
        return {
            "chart_type": "scatter",
            "title": title,
            "data": {
                "x":      [self._to_num(r.get(xc, 0)) for r in rows],
                "y":      [self._to_num(r.get(yc, 0)) for r in rows],
                "labels": [str(r.get(lc, "")) for r in rows] if lc else []
            },
            **layout
        }

    # ── Heatmap ───────────────────────────────────────────────────────────────
    def _heatmap_widget(self, title, rows, cols, layout):
        if len(cols) < 3:
            return self._table_widget(title, rows, cols, {"w": 12, "h": 5})
        rc, cc, vc = cols[0], cols[1], cols[2]
        row_labels = sorted(set(str(r.get(rc) if r.get(rc) is not None else "Unknown") for r in rows))
        col_labels = sorted(set(str(r.get(cc) if r.get(cc) is not None else "Unknown") for r in rows))
        matrix = {
            (
                str(r.get(rc) if r.get(rc) is not None else "Unknown"),
                str(r.get(cc) if r.get(cc) is not None else "Unknown"),
            ): self._to_num(r.get(vc, 0))
            for r in rows
        }
        z = [[matrix.get((rl, cl), 0) for cl in col_labels] for rl in row_labels]
        return {
            "chart_type": "heatmap",
            "title": title,
            "data": {"y_labels": row_labels, "x_labels": col_labels, "values": z},
            **layout
        }

    # ── Radar ─────────────────────────────────────────────────────────────────
    def _radar_widget(self, title, rows, cols, layout):
        label_col, val_col = self._detect_label_value(cols, rows)
        labels = [str(r.get(label_col, "?")) for r in rows]
        values = [self._to_num(r.get(val_col, 0)) for r in rows]
        return {
            "chart_type": "radar",
            "title": title,
            "data": {"labels": labels, "values": values},
            **layout
        }

    # ─────────────────────────────────────────────────────────────────────────
    def generate_text_summary(self, query_results: list[dict],
                              user_prompt: str) -> str:
        """
        Generate a concise prose summary of results using Groq.
        Called by the Orchestrator for text-only or summary responses.
        """
        # Build a compact data snapshot
        snippets = []
        for qr in query_results:
            rows = qr.get("rows", [])
            if not rows:
                continue
            snippet = f"{qr['label']}: {rows[:5]}"
            snippets.append(snippet)

        data_str = "\n".join(snippets) or "No data available."

        system = (
            "You are a CRM analytics assistant. Given data results, write a SHORT, "
            "professional summary (max 120 words). Use bold numbers. "
            "No steps, no JSON, no 'based on the data'."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Request: {user_prompt}\n\nData:\n{data_str}"}
        ]
        try:
            completion = self._groq.chat.completions.create(
                model=GROQ_MODEL, messages=messages,
                temperature=0.3, max_tokens=300
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"[VizAgent] Summary error: {e}")
            return ""

    # ────────── helpers ────────────────────────────────────────────────────────

    def _detect_label_value(self, cols: list, rows: list) -> tuple[str, str]:
        """
        Heuristically identify the label column and numeric value column.

        Strategy:
          1. The label column is the first non-numeric column.
          2. The value column is the first numeric column that is not the label.
          3. If everything is numeric, col[0]=label, col[1]=value.
          4. If only one column, use it for both.
        """
        if not cols:
            return ("label", "value")
        if len(cols) == 1:
            return (cols[0], cols[0])

        # Identify non-numeric and numeric columns
        non_numeric = [c for c in cols if not self._col_is_numeric(c, rows)]
        numeric     = [c for c in cols if     self._col_is_numeric(c, rows)]

        label_col = non_numeric[0] if non_numeric else cols[0]
        value_col = next((c for c in numeric if c != label_col), None)

        # Fallback: if no distinct numeric col found, use the second column
        if value_col is None:
            value_col = cols[1] if len(cols) > 1 and cols[1] != label_col else cols[0]

        return (label_col, value_col)

    def _is_data_sufficient(self, otype: str, rows: list[dict], cols: list[str]) -> bool:
        """Return True only when data quality is adequate for a useful chart."""
        if otype in ("kpi", "numeric", "gauge", "dial", "bullet", "table", "pivot", "report"):
            return bool(rows)

        if len(rows) < 2:
            return False

        numeric_cols = [c for c in cols if self._col_is_numeric(c, rows)]
        non_numeric_cols = [c for c in cols if c not in numeric_cols]

        def distinct_count(col: str) -> int:
            vals = {str(r.get(col)) for r in rows if r.get(col) not in (None, "")}
            return len(vals)

        # Pie/Funnel family should have meaningful category spread
        if otype in ("pie", "doughnut", "donut", "ring", "halfpie", "halfring", "halfdoughnut", "funnel"):
            if not numeric_cols:
                return False
            label_col = non_numeric_cols[0] if non_numeric_cols else cols[0]
            return distinct_count(label_col) >= 2

        # Cartesian bars/lines should have label variety and at least one numeric metric
        if otype in ("bar", "column", "horizontalBar", "line", "area", "histogram"):
            if not numeric_cols:
                return False
            label_col = non_numeric_cols[0] if non_numeric_cols else cols[0]
            return distinct_count(label_col) >= 2

        # Multi-series charts need at least 2 numeric series
        if otype in ("stacked", "stackedbar", "groupedBar", "multiLine", "combination", "combo"):
            return len(numeric_cols) >= 2 and len(rows) >= 2

        # Scatter/bubble need richer numeric data
        if otype == "scatter":
            return len(numeric_cols) >= 2 and len(rows) >= 3
        if otype in ("bubble", "packedbubble"):
            return len(numeric_cols) >= 1 and len(rows) >= 3

        # Treemap/sunburst/radar/heatmap and others: require at least one numeric and two rows
        if otype in ("treemap", "sunburst", "radar", "spider", "web", "heatmap", "polarArea", "box", "violin", "waterfall", "butterfly"):
            return len(numeric_cols) >= 1 and len(rows) >= 2

        return len(numeric_cols) >= 1 and len(rows) >= 2

    def _col_is_numeric(self, col: str, rows: list) -> bool:
        """
        Return True if the majority of non-null values in *col* are numeric.
        Handles both raw strings (from CSV) and already-typed int/float values
        (from ValidationAgent sanitization).
        """
        sample = [r.get(col) for r in rows[:20] if r.get(col) is not None]
        if not sample:
            return False
        numeric = sum(
            1 for v in sample
            if isinstance(v, (int, float))
            or (isinstance(v, str) and re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$", v.replace(",", "").strip()))
        )
        return numeric > len(sample) / 2

    @staticmethod
    def _looks_numeric(s: str) -> bool:
        """Quick check — is this string likely a number (after stripping symbols)?"""
        if s is None:
            return False
        cleaned = re.sub(r"[^\d.\-+eE]", "", str(s).strip())
        return bool(cleaned) and bool(re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$", cleaned))

    @staticmethod
    def _to_num(val) -> "float | int":
        """
        Coerce a value to int / float.

        Handles:
          • Already-typed int / float       → returned as-is
          • None                             → 0
          • "1,234"                          → 1234
          • "$1,234.56" / "€50"             → 1234.56 / 50
          • "45%"                            → 45.0
          • "1.5K" / "2M" / "3B"           → 1500 / 2000000 / 3000000000
        Returns 0 if conversion is impossible (never raises).
        """
        if isinstance(val, (int, float)):
            return val
        if val is None:
            return 0

        s = str(val).strip()

        # Percentage
        if s.endswith("%"):
            try:
                return float(s[:-1])
            except ValueError:
                pass

        # K / M / B suffixes
        upper = s.upper()
        for suffix, mult in (("K", 1_000), ("M", 1_000_000), ("B", 1_000_000_000)):
            if upper.endswith(suffix):
                try:
                    result = float(upper[:-1].replace(",", "")) * mult
                    return int(result) if result == int(result) else result
                except ValueError:
                    pass

        # Strip all non-numeric chars, preserving a leading minus
        leading_minus = s.startswith("-")
        stripped = re.sub(r"[^\d.\-+eE]", "", s)
        if not stripped or stripped in ("-", "+", "."):
            return 0
        if leading_minus and not stripped.startswith("-"):
            stripped = "-" + stripped

        try:
            f = float(stripped)
            return int(f) if f == int(f) else f
        except (ValueError, OverflowError):
            return 0

    @staticmethod
    def _format_kpi_number(num: "float | int") -> str:
        """
        Format a numeric KPI value for display.
          1234567  → "1,234,567"
          1234567.89 → "1,234,567.89"
          0.456    → "0.456"
        """
        if isinstance(num, int) or (isinstance(num, float) and num == int(num)):
            return f"{int(num):,}"
        return f"{num:,.2f}".rstrip("0").rstrip(".")

    @staticmethod
    def _kpi_icon(title: str) -> str:
        t = title.lower()
        if any(w in t for w in ("lead",)):                      return "🎯"
        if any(w in t for w in ("deal", "pipeline", "stage")):  return "💼"
        if any(w in t for w in ("revenue", "amount", "money",
                                "value", "earn")):               return "💰"
        if any(w in t for w in ("contact",)):                    return "👤"
        if any(w in t for w in ("account",)):                    return "🏢"
        if any(w in t for w in ("task",)):                       return "✅"
        if any(w in t for w in ("call",)):                       return "📞"
        if any(w in t for w in ("event",)):                      return "📅"
        if any(w in t for w in ("won", "closed won")):           return "🏆"
        if any(w in t for w in ("lost",)):                       return "❌"
        return "📊"

    @staticmethod
    def _error_widget(title: str, error: str) -> dict:
        return {
            "chart_type": "kpi",
            "title": title,
            "data": {"value": "Error", "subtitle": error[:80], "icon": "⚠️"},
            "w": 4, "h": 2
        }

    @staticmethod
    def _empty_widget(title: str, otype: str) -> dict:
        return {
            "chart_type": "kpi",
            "title": title,
            "data": {"value": "No Data", "subtitle": "Query returned 0 rows", "icon": "📭"},
            "w": 3, "h": 2
        }
