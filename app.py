"""
Streamlit dashboard for the Mental Health Burnout dataset.

Four pages:
    1. Overview & Data Explorer
    2. EDA & Visualizations
    3. Model Performance
    4. Predict Burnout (live)

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from feature_engineering import engineer_features
from models import SoftVotingEnsemble  # noqa: F401 — must be imported so joblib can unpickle model.joblib

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "mental_health_burnout_tech_2026.csv"
ART_DIR = HERE / "artifacts"

CLASS_ORDER = ["Low", "Moderate", "High", "Severe"]
CLASS_COLORS = {
    "Low": "#2ecc71",
    "Moderate": "#f1c40f",
    "High": "#e67e22",
    "Severe": "#e74c3c",
}

st.set_page_config(
    page_title="Tech Burnout Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(ART_DIR / "model.joblib")


@st.cache_resource(show_spinner=False)
def load_label_encoder():
    return joblib.load(ART_DIR / "label_encoder.joblib")


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads((ART_DIR / "metrics.json").read_text())


@st.cache_data(show_spinner=False)
def load_feature_meta() -> dict:
    return json.loads((ART_DIR / "feature_meta.json").read_text())


def artifacts_present() -> bool:
    needed = ["model.joblib", "label_encoder.joblib", "metrics.json", "feature_meta.json"]
    return all((ART_DIR / f).exists() for f in needed)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def group_features(numeric_names: list[str], categorical_names: list[str]) -> dict[str, list[str]]:
    """Group inputs for the Predict form so it isn't a wall of 25 controls."""
    groups: dict[str, list[str]] = {
        "Demographics": [],
        "Work Context": [],
        "Lifestyle": [],
        "Subjective Scores": [],
        "Support / Tools": [],
    }
    demographics = {"age", "gender", "country"}
    work = {
        "job_role", "seniority_level", "years_experience", "years_at_company",
        "company_size", "industry", "work_mode", "salary_usd",
        "work_hours_per_week", "meetings_per_day", "team_size",
    }
    lifestyle = {"sleep_hours_per_night", "exercise_days_per_week", "vacation_days_taken"}
    support = {"therapy_access", "uses_therapy", "ai_tools_daily"}

    for name in numeric_names + categorical_names:
        if name in demographics:
            groups["Demographics"].append(name)
        elif name in work:
            groups["Work Context"].append(name)
        elif name in lifestyle:
            groups["Lifestyle"].append(name)
        elif name in support:
            groups["Support / Tools"].append(name)
        else:
            groups["Subjective Scores"].append(name)
    return groups


def color_badge(label: str) -> str:
    color = CLASS_COLORS.get(label, "#7f8c8d")
    return (
        f"<div style='background:{color};color:white;padding:14px;border-radius:10px;"
        f"text-align:center;font-size:30px;font-weight:700;'>{label}</div>"
    )


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview(df: pd.DataFrame) -> None:
    st.title("Overview & Data Explorer")
    st.caption("Filter the dataset and inspect summary statistics.")

    # Sidebar filters
    st.sidebar.header("Filters")
    countries = sorted(df["country"].unique())
    roles = sorted(df["job_role"].unique())
    modes = sorted(df["work_mode"].unique())

    sel_countries = st.sidebar.multiselect("Country", countries, default=countries)
    sel_roles = st.sidebar.multiselect("Job role", roles, default=roles)
    sel_modes = st.sidebar.multiselect("Work mode", modes, default=modes)
    age_range = st.sidebar.slider(
        "Age range", int(df["age"].min()), int(df["age"].max()),
        (int(df["age"].min()), int(df["age"].max())),
    )
    hours_range = st.sidebar.slider(
        "Work hours per week",
        int(df["work_hours_per_week"].min()),
        int(df["work_hours_per_week"].max()),
        (int(df["work_hours_per_week"].min()), int(df["work_hours_per_week"].max())),
    )

    filtered = df[
        df["country"].isin(sel_countries)
        & df["job_role"].isin(sel_roles)
        & df["work_mode"].isin(sel_modes)
        & df["age"].between(*age_range)
        & df["work_hours_per_week"].between(*hours_range)
    ]

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Employees", f"{len(filtered):,}")
    severe_pct = (filtered["burnout_level"] == "Severe").mean() * 100 if len(filtered) else 0
    c2.metric("% Severe burnout", f"{severe_pct:.1f}%")
    c3.metric("Avg work hours/wk", f"{filtered['work_hours_per_week'].mean():.1f}" if len(filtered) else "-")
    c4.metric("Avg sleep hrs/night", f"{filtered['sleep_hours_per_night'].mean():.2f}" if len(filtered) else "-")
    c5.metric("Avg job satisfaction", f"{filtered['job_satisfaction_score'].mean():.2f}" if len(filtered) else "-")

    st.markdown("### Sample of filtered data")
    st.caption(f"Showing first 200 of {len(filtered):,} rows")
    st.dataframe(filtered.head(200), use_container_width=True)

    st.download_button(
        "Download filtered data (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_burnout.csv",
        mime="text/csv",
    )


def page_eda(df: pd.DataFrame) -> None:
    st.title("EDA & Visualizations")

    # 1. Burnout level distribution
    st.subheader("Burnout level distribution")
    counts = df["burnout_level"].value_counts().reindex(CLASS_ORDER).reset_index()
    counts.columns = ["burnout_level", "count"]
    fig = px.bar(
        counts, x="burnout_level", y="count",
        color="burnout_level", color_discrete_map=CLASS_COLORS,
        category_orders={"burnout_level": CLASS_ORDER},
    )
    st.plotly_chart(fig, use_container_width=True)

    # 2. Burnout by group
    st.subheader("Burnout level by group")
    group_options = ["country", "job_role", "work_mode", "industry", "seniority_level", "gender", "company_size"]
    col_a, col_b = st.columns([3, 1])
    group_col = col_a.selectbox("Group by", group_options, index=1)
    normalize = col_b.checkbox("100% stacked", value=True)

    grp = df.groupby([group_col, "burnout_level"]).size().reset_index(name="count")
    if normalize:
        totals = grp.groupby(group_col)["count"].transform("sum")
        grp["share"] = grp["count"] / totals
        y_col, y_label = "share", "Share"
    else:
        y_col, y_label = "count", "Count"
    fig = px.bar(
        grp, x=group_col, y=y_col, color="burnout_level",
        color_discrete_map=CLASS_COLORS,
        category_orders={"burnout_level": CLASS_ORDER},
        labels={y_col: y_label},
    )
    fig.update_layout(barmode="stack", xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    # 3. Numeric feature histogram
    st.subheader("Numeric feature distribution")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "employee_id"]
    col_a, col_b = st.columns([3, 1])
    num_col = col_a.selectbox("Feature", numeric_cols, index=numeric_cols.index("stress_score"))
    split = col_b.checkbox("Split by burnout level", value=True)
    fig = px.histogram(
        df, x=num_col, nbins=40,
        color="burnout_level" if split else None,
        color_discrete_map=CLASS_COLORS,
        category_orders={"burnout_level": CLASS_ORDER},
        barmode="overlay" if split else "relative",
        opacity=0.7 if split else 1.0,
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4. Correlation heatmap
    st.subheader("Correlation heatmap (numeric features)")
    corr = df[numeric_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=False, color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # 5. Box plot
    st.subheader("Box plot by burnout level")
    box_col = st.selectbox(
        "Numeric feature for box plot", numeric_cols,
        index=numeric_cols.index("work_hours_per_week"),
        key="box_select",
    )
    fig = px.box(
        df, x="burnout_level", y=box_col, color="burnout_level",
        color_discrete_map=CLASS_COLORS,
        category_orders={"burnout_level": CLASS_ORDER},
    )
    st.plotly_chart(fig, use_container_width=True)


def page_model(metrics: dict) -> None:
    st.title("Model Performance")

    st.markdown(f"**Winning model:** `{metrics['winning_model']}`")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    c2.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
    c3.metric("Weighted F1", f"{metrics['weighted_f1']:.3f}")

    # Leaderboard
    st.subheader("Model comparison (held-out test set)")
    leaderboard = pd.DataFrame(metrics["leaderboard"]).T.reset_index().rename(columns={"index": "model"})
    st.dataframe(leaderboard.style.format({"macro_f1": "{:.3f}", "accuracy": "{:.3f}",
                                            "fit_seconds": "{:.1f}"}),
                 use_container_width=True)

    # Confusion matrix
    st.subheader("Confusion matrix")
    labels = metrics["class_labels"]
    cm = np.array(metrics["confusion_matrix"])
    fig = px.imshow(
        cm, x=labels, y=labels, text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "True", "color": "Count"},
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Per-class report
    st.subheader("Per-class precision / recall / F1")
    rep = metrics["classification_report"]
    rows = []
    for label in labels:
        r = rep[label]
        rows.append({
            "class": label,
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1-score"],
            "support": int(r["support"]),
        })
    st.dataframe(
        pd.DataFrame(rows).style.format(
            {"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}
        ),
        use_container_width=True,
    )

    # Permutation importance
    st.subheader("Top 15 features by permutation importance")
    imp = pd.DataFrame(metrics["permutation_importance"][:15])
    fig = px.bar(
        imp.sort_values("importance"),
        x="importance", y="feature", orientation="h",
        error_x="std",
        labels={"importance": "Drop in macro-F1 when shuffled", "feature": ""},
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def page_predict(model, label_encoder, feature_meta: dict, df: pd.DataFrame) -> None:
    st.title("Predict Burnout")
    st.caption(
        "Set the values below to describe a hypothetical employee. "
        "Click **Predict** to see the model's burnout-level estimate."
    )

    numeric_names = list(feature_meta["numeric"].keys())
    categorical_names = list(feature_meta["categorical"].keys())
    iqr_bounds = feature_meta.get("iqr_bounds", {})

    groups = group_features(numeric_names, categorical_names)
    inputs: dict[str, object] = {}

    with st.form("predict_form"):
        for group_name, members in groups.items():
            if not members:
                continue
            with st.expander(group_name, expanded=(group_name in {"Work Context", "Lifestyle"})):
                cols = st.columns(2)
                for i, name in enumerate(members):
                    target_col = cols[i % 2]
                    if name in feature_meta["numeric"]:
                        meta = feature_meta["numeric"][name]
                        lo, hi, med = meta["min"], meta["max"], meta["median"]
                        if meta["is_integer"]:
                            inputs[name] = target_col.slider(
                                name, int(lo), int(hi), int(med), step=1,
                            )
                        else:
                            inputs[name] = target_col.slider(
                                name, float(lo), float(hi), float(med), step=0.1,
                            )
                    else:
                        options = feature_meta["categorical"][name]
                        inputs[name] = target_col.selectbox(name, options, index=0)
        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        return

    # Build single-row frame from raw inputs, apply engineering, then select
    # the exact columns the model was trained on (feature_order).
    raw_row = pd.DataFrame([inputs])
    engineered_row = engineer_features(raw_row)
    feature_order = feature_meta["feature_order"]
    X_row = engineered_row[feature_order]

    proba = model.predict_proba(X_row)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = label_encoder.classes_[pred_idx]

    st.markdown("### Prediction")
    st.markdown(color_badge(pred_label), unsafe_allow_html=True)

    # Probability bar chart
    proba_df = pd.DataFrame({
        "burnout_level": label_encoder.classes_,
        "probability": proba,
    })
    proba_df["burnout_level"] = pd.Categorical(proba_df["burnout_level"], categories=CLASS_ORDER, ordered=True)
    proba_df = proba_df.sort_values("burnout_level")
    fig = px.bar(
        proba_df, x="burnout_level", y="probability",
        color="burnout_level", color_discrete_map=CLASS_COLORS,
        text=proba_df["probability"].apply(lambda v: f"{v:.1%}"),
    )
    fig.update_layout(showlegend=False, yaxis_tickformat=".0%", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    # Out-of-range warnings using saved IQR bounds.
    warnings = []
    for col, b in iqr_bounds.items():
        if col in inputs:
            v = inputs[col]
            if v < b["low"] or v > b["high"]:
                warnings.append(f"- `{col}` = {v} is outside the typical training range "
                                f"[{b['low']:.1f}, {b['high']:.1f}]")
    if warnings:
        with st.expander("Inputs outside typical range"):
            st.markdown("\n".join(warnings))

    # Quick interpretation using per-class medians of the data.
    st.markdown("### Why this prediction?")
    notable = []
    medians_by_class = df.groupby("burnout_level")[[
        "stress_score", "work_hours_per_week", "sleep_hours_per_night",
        "job_satisfaction_score", "work_life_balance_score",
    ]].median()
    if pred_label in medians_by_class.index:
        for col in medians_by_class.columns:
            target_med = medians_by_class.loc[pred_label, col]
            v = inputs.get(col)
            if v is None:
                continue
            # Closeness threshold: 10% of feature range.
            rng = df[col].max() - df[col].min()
            if rng and abs(v - target_med) < 0.15 * rng:
                notable.append(f"- `{col}` ({v}) is close to the median for **{pred_label}** burnout ({target_med:.1f})")
    if notable:
        st.markdown("\n".join(notable))
    else:
        st.caption("Top-driving features (per the permutation importance on the Model page) include "
                   "stress_score, work_hours_per_week, sleep_hours_per_night and vacation_days_taken.")


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
def main() -> None:
    st.sidebar.title("Tech Burnout Dashboard")
    page = st.sidebar.radio(
        "Page",
        ["Overview", "EDA", "Model Performance", "Predict"],
    )

    if not artifacts_present():
        st.error(
            "Model artifacts not found. Run `python train_model.py` first to "
            "generate the `artifacts/` directory."
        )
        st.stop()

    df = load_data()
    if page == "Overview":
        page_overview(df)
    elif page == "EDA":
        page_eda(df)
    elif page == "Model Performance":
        page_model(load_metrics())
    elif page == "Predict":
        page_predict(load_model(), load_label_encoder(), load_feature_meta(), df)


if __name__ == "__main__":
    main()
