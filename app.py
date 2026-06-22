from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from dicom_parser import (
    classify_rt_files,
    extract_plan_summary,
    extract_structures,
    get_prescription_dose_gy,
    load_dicoms,
    save_uploaded_files,
)
from dvh_engine import calculate_dvh_metrics, dvh_note, global_hotspot_analysis
from scorecard_engine import build_metric_table, domain_scores, final_grade
from spider_chart import make_overlay_spider_chart, make_spider_chart, make_structure_overlay_chart


st.set_page_config(page_title="DTI - HN SPIDERplan Scorecard", layout="wide")


@st.cache_data
def load_config() -> dict:
    config_path = Path(__file__).parent / "scoring_config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _is_excluded_structure_name(name: str) -> bool:
    return str(name).strip().lower().startswith("z")


def _is_ln_helper_structure(name: str) -> bool:
    n = str(name).strip().lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", n) if t]
    return "ln" in tokens or n.startswith("ln") or n.endswith("ln")


def _is_target_name(name: str) -> bool:
    n = str(name).strip().lower()
    if _is_excluded_structure_name(name) or _is_ln_helper_structure(name):
        return False
    if n.endswith("opti"):
        return False
    return any(k in n for k in ["ptv", "ctv", "gtv"])


def _is_eval_structure(name: str) -> bool:
    return str(name).strip().lower().endswith("_eval")


def _parent_from_eval_name(name: str) -> str:
    return re.sub(r"_eval$", "", str(name).strip(), flags=re.IGNORECASE)


def _name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _target_type(name: str) -> str:
    n = str(name).lower()
    suffix = "_eval" if _is_eval_structure(name) else ""
    if "ptv" in n:
        return f"PTV{suffix}"
    if "ctv" in n:
        return f"CTV{suffix}"
    if "gtv" in n:
        return f"GTV{suffix}"
    return f"Target{suffix}"


def _rx_values_from_plan(rtplan) -> list[float]:
    doses: list[float] = []
    for ref in getattr(rtplan, "DoseReferenceSequence", []) or []:
        dose = getattr(ref, "TargetPrescriptionDose", None)
        if dose is None:
            continue
        try:
            d = float(dose)
            if d > 0 and d not in doses:
                doses.append(d)
        except Exception:
            pass
    return sorted(doses, reverse=True)


def _infer_rx_from_name(name: str, known_rx: list[float]) -> tuple[float | None, str]:
    base = _parent_from_eval_name(name) if _is_eval_structure(name) else str(name)
    n_raw = base.lower()
    n = n_raw.replace("cgy", " cgy").replace("gy", " gy")

    m = re.search(r"(\d{3,5}(?:\.\d+)?)\s*cgy", n)
    if m:
        return float(m.group(1)) / 100.0, "name cGy"

    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*gy", n)
    if m:
        return float(m.group(1)), "name Gy"

    m = re.search(r"(?:ptv|ctv|gtv)[_\-\s]*(?:high|mid|low|boost)?[_\-\s]*(\d{2,5}(?:\.\d+)?)", n)
    if not m:
        m = re.search(r"(\d{2,5}(?:\.\d+)?)", n)
    if m:
        raw = float(m.group(1))
        value = raw / 100.0 if raw >= 1000 else raw
        if 10 <= value <= 120:
            return value, "name pattern"

    if "gtv" in n_raw and known_rx:
        return max(known_rx), "GTV assigned highest plan Rx"

    if len(known_rx) == 1:
        return known_rx[0], "single plan Rx"
    return None, "manual required"


def build_target_rx_table(structure_df: pd.DataFrame, rtplan) -> pd.DataFrame:
    known_rx = _rx_values_from_plan(rtplan)
    cols = ["structure", "target_type", "parent_structure", "assigned_rx_gy", "rx_source", "scoring_role", "include_in_score"]
    if structure_df is None or structure_df.empty or "structure_name" not in structure_df.columns:
        return pd.DataFrame(columns=cols)

    target_names = [str(x) for x in structure_df["structure_name"].tolist() if _is_target_name(str(x))]
    all_name_keys = {_name_key(n): n for n in target_names}
    rows: list[dict] = []
    base_rx_by_key: dict[str, float] = {}

    for name in target_names:
        if _is_eval_structure(name):
            continue
        rx, source = _infer_rx_from_name(name, known_rx)
        if rx is not None:
            base_rx_by_key[_name_key(name)] = rx
        rows.append({
            "structure": name,
            "target_type": _target_type(name),
            "parent_structure": "",
            "assigned_rx_gy": rx,
            "rx_source": source,
            "scoring_role": "Coverage / target-dose quality",
            "include_in_score": True,
        })

    for name in target_names:
        if not _is_eval_structure(name):
            continue
        parent = _parent_from_eval_name(name)
        parent_key = _name_key(parent)
        parent_display = all_name_keys.get(parent_key, parent)
        rx = base_rx_by_key.get(parent_key)
        source = "inherited from parent target"
        if rx is None:
            rx, source = _infer_rx_from_name(parent, known_rx)
            if rx is not None:
                source = "inferred from parent name"
        rows.append({
            "structure": name,
            "target_type": _target_type(name),
            "parent_structure": parent_display,
            "assigned_rx_gy": rx,
            "rx_source": source if rx is not None else "manual required - parent Rx missing",
            "scoring_role": "V105% hotspot review only",
            "include_in_score": True,
        })

    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out["sort_key"] = out["structure"].str.lower().str.replace("_eval", "zz_eval", regex=False)
        out = out.sort_values(["sort_key", "scoring_role"]).drop(columns=["sort_key"]).reset_index(drop=True)
    return out


def _status_from_score(score: Any) -> str:
    try:
        s = float(score)
    except Exception:
        return "not_scored"
    if s >= 90:
        return "passed"
    if s >= 50:
        return "marginal"
    return "failed"


def _metric_status_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["status"] = out.get("score", pd.Series([None] * len(out))).apply(_status_from_score)
    return out


def _highlight_scorecard(row):
    status = row.get("status", "not_scored")
    color = ""
    if status == "passed":
        color = "background-color: #d1fae5; color: #064e3b;"
    elif status == "marginal":
        color = "background-color: #fef3c7; color: #78350f;"
    elif status == "failed":
        color = "background-color: #fee2e2; color: #7f1d1d;"
    return [color for _ in row]


def _display_highlighted(df: pd.DataFrame, **kwargs):
    if df is None or df.empty:
        st.info("No rows to display.")
        return
    styled = df.style.apply(_highlight_scorecard, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True, **kwargs)


def _process_plan(plan_label: str, uploaded_files, structure_limit: int, require_rx: bool, editor_key: str) -> dict | None:
    if not uploaded_files:
        return None

    uploaded_paths = save_uploaded_files(uploaded_files, upload_dir=Path(f"uploaded_dicoms_{editor_key}"))
    dicoms = load_dicoms(uploaded_paths)
    if not dicoms:
        st.error(f"{plan_label}: no readable DICOM files were found.")
        return None

    files = classify_rt_files(dicoms)
    rtplan = files.get("RTPLAN")
    rtstruct = files.get("RTSTRUCT")
    rtdose = files.get("RTDOSE")
    other_files = files.get("OTHER", [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{plan_label} RP", "Yes" if rtplan is not None else "No")
    c2.metric(f"{plan_label} RS", "Yes" if rtstruct is not None else "No")
    c3.metric(f"{plan_label} RD", "Yes" if rtdose is not None else "No")
    c4.metric(f"{plan_label} Other", len(other_files))

    missing = []
    if rtplan is None:
        missing.append("RP")
    if rtstruct is None:
        missing.append("RS")
    if rtdose is None:
        missing.append("RD")
    if missing:
        st.error(f"{plan_label}: Full scoring requires " + ", ".join(missing) + ".")
        return None

    plan_summary = extract_plan_summary(rtplan)
    rx_dose_gy = get_prescription_dose_gy(rtplan)
    structures = extract_structures(rtstruct)
    structure_df = pd.DataFrame(structures)

    st.markdown(f"#### {plan_label} Target Prescription Assignment")
    target_rx_df = build_target_rx_table(structure_df, rtplan)
    if target_rx_df.empty:
        st.error(f"{plan_label}: no PTV/CTV/GTV structures were detected.")
        return None

    edited_target_rx_df = st.data_editor(
        target_rx_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "structure": st.column_config.TextColumn("Structure", disabled=True),
            "target_type": st.column_config.TextColumn("Target type", disabled=True),
            "parent_structure": st.column_config.TextColumn("Parent target", disabled=True),
            "assigned_rx_gy": st.column_config.NumberColumn("Assigned Rx Gy", min_value=0.0, max_value=120.0, step=0.1, format="%.1f"),
            "rx_source": st.column_config.TextColumn("Rx source", disabled=True),
            "scoring_role": st.column_config.TextColumn("Scoring role", disabled=True),
            "include_in_score": st.column_config.CheckboxColumn("Score", default=True),
        },
        key=f"target_rx_editor_{editor_key}",
    )

    active_rx_df = edited_target_rx_df[edited_target_rx_df["include_in_score"] == True].copy()
    active_rx_df["assigned_rx_gy"] = pd.to_numeric(active_rx_df["assigned_rx_gy"], errors="coerce")
    missing_rx = active_rx_df[active_rx_df["assigned_rx_gy"].isna() | (active_rx_df["assigned_rx_gy"] <= 0)]
    if not missing_rx.empty and require_rx:
        st.warning(f"{plan_label}: some scored targets/eval structures do not have an assigned Rx.")
        st.dataframe(missing_rx[["structure", "target_type", "parent_structure", "assigned_rx_gy", "rx_source", "scoring_role"]], use_container_width=True, hide_index=True)
        return None

    target_rx_map = {
        str(row["structure"]): float(row["assigned_rx_gy"])
        for _, row in active_rx_df.iterrows()
        if pd.notna(row["assigned_rx_gy"]) and float(row["assigned_rx_gy"]) > 0
    }
    eval_names = [s for s in target_rx_map if _is_eval_structure(s)]
    priority_structures = list(target_rx_map.keys())

    with st.spinner(f"Calculating {plan_label} DVH metrics and scorecard..."):
        dvh_df, dvh_warnings = calculate_dvh_metrics(
            rtstruct=rtstruct,
            rtdose=rtdose,
            rx_dose_gy=None,
            structure_limit=int(structure_limit),
            rx_map=target_rx_map,
            priority_structures=priority_structures,
        )

    if dvh_df is None or dvh_df.empty:
        st.error(f"{plan_label}: no DVH metrics could be calculated.")
        return None

    metric_df = build_metric_table(dvh_df, rx_dose_gy=rx_dose_gy)
    global_hotspot_df, global_hotspot_warnings = global_hotspot_analysis(rtstruct, rtdose, rx_map=target_rx_map)
    if global_hotspot_warnings:
        dvh_warnings.extend(global_hotspot_warnings)
    if global_hotspot_df is not None and not global_hotspot_df.empty:
        metric_df = pd.concat([metric_df, global_hotspot_df], ignore_index=True)

    metric_df = _metric_status_df(metric_df)
    domain_df = domain_scores(metric_df)
    grade = final_grade(domain_df)

    return {
        "label": plan_label,
        "plan_summary": pd.DataFrame([plan_summary]) if plan_summary else pd.DataFrame(),
        "structure_df": structure_df,
        "target_rx_df": edited_target_rx_df,
        "dvh_df": dvh_df,
        "metric_df": metric_df,
        "domain_df": domain_df,
        "grade": grade,
        "fig": make_spider_chart(domain_df, name=plan_label),
        "eval_names": eval_names,
        "warnings": dvh_warnings,
    }


def _grade_style(score_a, score_b):
    try:
        a = float(score_a)
        b = float(score_b)
    except Exception:
        return ""
    if a > b:
        return "background-color: #d1fae5; color: #064e3b; font-weight: 700;"
    if a < b:
        return "background-color: #fee2e2; color: #7f1d1d;"
    return "background-color: #f3f4f6;"


def _comparison_summary(plan_a: dict, plan_b: dict) -> pd.DataFrame:
    rows = []
    for p in [plan_a, plan_b]:
        g = p["grade"]
        rows.append({"plan": p["label"], "score": g.get("score", 0), "grade": g.get("grade", "N/A")})
    return pd.DataFrame(rows)


config = load_config()

st.title("DTI - HN SPIDERplan Scorecard")
st.caption("Single-plan scoring plus optional two-plan comparison with overlapping SPIDERplan, OAR, and target-volume scorecards.")

with st.expander("Clinical / security disclaimer", expanded=False):
    st.write(
        "This prototype is for research, development, and local plan-review support only. "
        "It is not a replacement for clinical TPS DVH review, physician approval, physicist QA, chart rounds, "
        "or institutional policy. Use only de-identified or institutionally approved datasets. "
        "Validate all DVH and scorecard outputs against Eclipse/ARIA or your clinical TPS before any clinical use."
    )

with st.sidebar:
    st.header("Options")
    structure_limit = st.number_input("Structure calculation limit", min_value=1, max_value=300, value=120, step=5)
    score_only_assigned_targets = st.checkbox("Require Rx for scored targets", value=True)
    st.markdown("---")
    st.caption("Upload Plan B to activate comparison mode.")

col_a, col_b = st.columns(2)
with col_a:
    plan_a_files = st.file_uploader(
        "Plan A: Upload RP + RS + RD files",
        type=["dcm", "dicom", "DCM"],
        accept_multiple_files=True,
        key="plan_a_upload",
    )
with col_b:
    plan_b_files = st.file_uploader(
        "Plan B: Upload RP + RS + RD files optional comparison",
        type=["dcm", "dicom", "DCM"],
        accept_multiple_files=True,
        key="plan_b_upload",
    )

if not plan_a_files:
    st.info("Upload Plan A RP + RS + RD files to generate the scorecard. Upload Plan B to compare two plans side by side.")
    st.stop()

try:
    st.markdown("---")
    st.header("Plan Processing")
    with st.expander("Plan A setup", expanded=True):
        plan_a = _process_plan("Plan A", plan_a_files, int(structure_limit), score_only_assigned_targets, "plan_a")
    if plan_a is None:
        st.stop()

    plan_b = None
    if plan_b_files:
        with st.expander("Plan B setup", expanded=True):
            plan_b = _process_plan("Plan B", plan_b_files, int(structure_limit), score_only_assigned_targets, "plan_b")
        if plan_b is None:
            st.warning("Plan B could not be scored. Plan A results are still available below.")

    st.markdown("---")
    st.header("SPIDERPlan Scorecard Snapshot")

    if plan_b is not None:
        a_score = float(plan_a["grade"].get("score", 0))
        b_score = float(plan_b["grade"].get("score", 0))
        a_style = _grade_style(a_score, b_score)
        b_style = _grade_style(b_score, a_score)
        gc1, gc2 = st.columns(2)
        with gc1:
            st.markdown(f"<div style='padding:1rem;border-radius:0.75rem;{a_style}'><h3>Plan A</h3><h1>{a_score:g}</h1><h2>Grade {plan_a['grade'].get('grade','N/A')}</h2></div>", unsafe_allow_html=True)
        with gc2:
            st.markdown(f"<div style='padding:1rem;border-radius:0.75rem;{b_style}'><h3>Plan B</h3><h1>{b_score:g}</h1><h2>Grade {plan_b['grade'].get('grade','N/A')}</h2></div>", unsafe_allow_html=True)
        st.plotly_chart(make_overlay_spider_chart(plan_a["domain_df"], plan_b["domain_df"], "Plan A", "Plan B", "Overall SPIDERplan comparison"), use_container_width=True)
    else:
        g = plan_a["grade"]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Final SPIDERPlan Score", f"{g.get('score', 0)}")
        s2.metric("Final SPIDERPlan Grade", str(g.get("grade", "N/A")))
        s3.metric("Scored Rows", len(plan_a["metric_df"][pd.to_numeric(plan_a["metric_df"].get("score"), errors="coerce").notna()]))
        s4.metric("PTV_eval Reviews", len(plan_a["eval_names"]))
        st.plotly_chart(plan_a["fig"], use_container_width=True)

    if plan_b is not None:
        st.markdown("---")
        st.header("Comparison SPIDERplan Graphs")
        t1, t2 = st.tabs(["Target volumes", "OARs"])
        with t1:
            fig_tv = make_structure_overlay_chart(plan_a["metric_df"], plan_b["metric_df"], "targets", "Plan A", "Plan B")
            if fig_tv is not None:
                st.plotly_chart(fig_tv, use_container_width=True)
            else:
                st.info("No target-volume score rows were available for comparison.")
        with t2:
            fig_oar = make_structure_overlay_chart(plan_a["metric_df"], plan_b["metric_df"], "oars", "Plan A", "Plan B")
            if fig_oar is not None:
                st.plotly_chart(fig_oar, use_container_width=True)
            else:
                st.info("No OAR score rows were available for comparison.")

    st.markdown("---")
    st.header("Final Metrics Table")
    st.caption("Color legend: green = passed/achieved, yellow = marginal, red = failed.")

    if plan_b is not None:
        tab_a, tab_b, tab_compare = st.tabs(["Plan A metrics", "Plan B metrics", "Side-by-side scores"])
        with tab_a:
            _display_highlighted(plan_a["metric_df"])
        with tab_b:
            _display_highlighted(plan_b["metric_df"])
        with tab_compare:
            a = plan_a["metric_df"].copy()
            b = plan_b["metric_df"].copy()
            for df, suffix in [(a, "A"), (b, "B")]:
                df["match_key"] = df.get("structure", "").astype(str) + " | " + df.get("scoring_role", "").astype(str)
                keep = ["match_key", "structure", "category", "scoring_role", "score", "grade", "status", "notes"]
                df.drop(columns=[c for c in df.columns if c not in keep], inplace=True)
                df.rename(columns={"score": f"score_{suffix}", "grade": f"grade_{suffix}", "status": f"status_{suffix}", "notes": f"notes_{suffix}"}, inplace=True)
            comp = pd.merge(a, b, on=["match_key", "structure", "category", "scoring_role"], how="outer")
            comp = comp.drop(columns=["match_key"])
            st.dataframe(comp, use_container_width=True, hide_index=True)
    else:
        _display_highlighted(plan_a["metric_df"])

    st.markdown("---")
    st.header("Detailed Review")
    tab_labels = ["Plan A"] + (["Plan B"] if plan_b is not None else [])
    tabs = st.tabs(tab_labels)
    for tab, p in zip(tabs, [plan_a] + ([plan_b] if plan_b is not None else [])):
        with tab:
            with st.expander("Domain scores", expanded=True):
                st.dataframe(p["domain_df"], use_container_width=True, hide_index=True)
            with st.expander("Target Rx assignment", expanded=False):
                st.dataframe(p["target_rx_df"], use_container_width=True, hide_index=True)
            with st.expander("Plan summary", expanded=False):
                st.dataframe(p["plan_summary"], use_container_width=True, hide_index=True)
            with st.expander("DVH / Dose Metrics", expanded=False):
                st.caption(dvh_note())
                st.dataframe(p["dvh_df"], use_container_width=True, hide_index=True)
            if p["warnings"]:
                with st.expander("DVH calculation warnings", expanded=False):
                    for warning in p["warnings"][:200]:
                        st.write(f"- {warning}")
                    if len(p["warnings"]) > 200:
                        st.write(f"...and {len(p['warnings']) - 200} additional warnings.")

    st.subheader("Export")
    e1, e2, e3 = st.columns(3)
    e1.download_button("Download Plan A Scorecard CSV", data=_csv_bytes(plan_a["metric_df"]), file_name="plan_a_spiderplan_metric_scorecard.csv", mime="text/csv")
    e2.download_button("Download Plan A DVH CSV", data=_csv_bytes(plan_a["dvh_df"]), file_name="plan_a_spiderplan_dvh_metrics.csv", mime="text/csv")
    if plan_b is not None:
        e3.download_button("Download Plan B Scorecard CSV", data=_csv_bytes(plan_b["metric_df"]), file_name="plan_b_spiderplan_metric_scorecard.csv", mime="text/csv")

except Exception as e:
    st.error("The app hit an unexpected error, but it was caught safely.")
    st.exception(e)
    st.stop()
