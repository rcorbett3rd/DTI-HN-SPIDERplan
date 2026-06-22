from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def _closed(labels, values):
    if not labels:
        return [], []
    return labels + [labels[0]], values + [values[0]]


def make_spider_chart(domain_df: pd.DataFrame, name: str = "SPIDERPlan"):
    if domain_df is None or domain_df.empty:
        return None
    labels = domain_df["domain"].astype(str).tolist()
    values = pd.to_numeric(domain_df["domain_score"], errors="coerce").fillna(0).astype(float).tolist()
    labels_closed, values_closed = _closed(labels, values)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values_closed, theta=labels_closed, fill="toself", name=name))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=520,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def make_overlay_spider_chart(domain_a: pd.DataFrame, domain_b: pd.DataFrame | None = None, name_a: str = "Plan A", name_b: str = "Plan B", title: str | None = None):
    if domain_a is None or domain_a.empty:
        return None
    domain_b = domain_b if domain_b is not None else pd.DataFrame()
    labels = sorted(set(domain_a.get("domain", pd.Series(dtype=str)).astype(str).tolist()) | set(domain_b.get("domain", pd.Series(dtype=str)).astype(str).tolist()))
    if not labels:
        return None

    def vals(df):
        if df is None or df.empty:
            return [0.0] * len(labels)
        lookup = dict(zip(df["domain"].astype(str), pd.to_numeric(df["domain_score"], errors="coerce").fillna(0)))
        return [float(lookup.get(x, 0.0)) for x in labels]

    fig = go.Figure()
    la, va = _closed(labels, vals(domain_a))
    fig.add_trace(go.Scatterpolar(r=va, theta=la, fill="toself", name=name_a, opacity=0.75))
    if domain_b is not None and not domain_b.empty:
        lb, vb = _closed(labels, vals(domain_b))
        fig.add_trace(go.Scatterpolar(r=vb, theta=lb, fill="toself", name=name_b, opacity=0.75))
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=560,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def make_structure_overlay_chart(metric_a: pd.DataFrame, metric_b: pd.DataFrame | None = None, filter_kind: str = "targets", name_a: str = "Plan A", name_b: str = "Plan B"):
    if metric_a is None or metric_a.empty:
        return None
    metric_b = metric_b if metric_b is not None else pd.DataFrame()

    def subset(df):
        if df is None or df.empty or "score" not in df.columns:
            return pd.DataFrame(columns=["structure", "score"])
        out = df.copy()
        out["score_numeric"] = pd.to_numeric(out["score"], errors="coerce")
        out = out[out["score_numeric"].notna()]
        cat = out.get("category", "").astype(str).str.lower()
        role = out.get("scoring_role", "").astype(str).str.lower()
        stname = out.get("structure", "").astype(str).str.lower()
        if filter_kind == "targets":
            mask = cat.str.contains("target|ptv_eval", regex=True) | stname.str.contains("ptv|ctv|gtv", regex=True)
        else:
            mask = cat.str.contains("oar|organ|serial|parallel", regex=True) | role.str.contains("oar|mean|dose|volume", regex=True)
            mask = mask & ~stname.str.contains("ptv|ctv|gtv|body|external", regex=True)
        out = out[mask]
        # If multiple rows share a structure, average the scored constraints for a structure-level radar point.
        return out.groupby("structure", as_index=False)["score_numeric"].mean().rename(columns={"score_numeric": "score"})

    sa = subset(metric_a)
    sb = subset(metric_b)
    labels = sorted(set(sa.get("structure", pd.Series(dtype=str)).astype(str).tolist()) | set(sb.get("structure", pd.Series(dtype=str)).astype(str).tolist()))
    if not labels:
        return None

    def vals(df):
        lookup = dict(zip(df["structure"].astype(str), pd.to_numeric(df["score"], errors="coerce").fillna(0))) if df is not None and not df.empty else {}
        return [float(lookup.get(x, 0.0)) for x in labels]

    fig = go.Figure()
    la, va = _closed(labels, vals(sa))
    fig.add_trace(go.Scatterpolar(r=va, theta=la, fill="toself", name=name_a, opacity=0.75))
    if metric_b is not None and not metric_b.empty:
        lb, vb = _closed(labels, vals(sb))
        fig.add_trace(go.Scatterpolar(r=vb, theta=lb, fill="toself", name=name_b, opacity=0.75))
    pretty = "Target volumes" if filter_kind == "targets" else "OARs"
    fig.update_layout(
        title=f"{pretty} comparison",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=620,
        margin=dict(l=40, r=40, t=70, b=40),
    )
    return fig
