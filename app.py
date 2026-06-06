from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from feature_engineering import engineer_features
from models import SoftVotingEnsemble

# joblib stored SoftVotingEnsemble under whatever module ran train_model.py.
# Register it under all plausible names so the stock unpickler resolves it.
for _mod_name in ("__main__", "models", "train_model"):
    _mod = sys.modules.setdefault(_mod_name, types.ModuleType(_mod_name))
    setattr(_mod, "SoftVotingEnsemble", SoftVotingEnsemble)

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Burnout Risk Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).resolve().parent
CSV_PATH  = HERE / "mental_health_burnout_tech_2026.csv"
ART_DIR   = HERE / "artifacts"

CLASS_ORDER  = ["Low", "Moderate", "High", "Severe"]
CLASS_COLORS = {"Low": "#27ae60", "Moderate": "#f39c12", "High": "#e67e22", "Severe": "#e74c3c"}
CLASS_EMOJI  = {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Severe": "🔴"}
CLASS_BG     = {"Low": "#eafaf1", "Moderate": "#fef9e7", "High": "#fef5e7", "Severe": "#fdf2f0"}

# ── Shared Plotly theme ───────────────────────────────────────────────────────
# Call apply_theme(fig) on every figure so every chart looks identical.
CHART_FONT   = dict(family="sans-serif", size=13, color="#2c3e50")
CHART_COLORS = ["#3498db", "#27ae60", "#e67e22", "#e74c3c",
                "#9b59b6", "#1abc9c", "#f39c12", "#2980b9"]

def apply_theme(fig: go.Figure, height: int = 400, title: str = "") -> go.Figure:
    """Apply a consistent light theme to every Plotly chart."""
    fig.update_layout(
        height=height,
        font=CHART_FONT,
        title=dict(text=title, font=dict(size=15, color="#2c3e50")) if title else {},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8f9fa",
        margin=dict(t=40 if title else 20, b=20, l=10, r=10),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#dee2e6",
            borderwidth=1,
            font=dict(color="#2c3e50"),
        ),
        xaxis=dict(
            gridcolor="#dee2e6", linecolor="#adb5bd",
            tickfont=dict(color="#495057"), title_font=dict(color="#2c3e50"),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="#dee2e6", linecolor="#adb5bd",
            tickfont=dict(color="#495057"), title_font=dict(color="#2c3e50"),
            zeroline=False,
        ),
        coloraxis_colorbar=dict(tickfont=dict(color="#2c3e50"),
                                title_font=dict(color="#2c3e50")),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS  — card layout, typography, colored badges
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global — force dark text on light background ── */
[data-testid="stAppViewContainer"] { background-color: #f8f9fa; }
[data-testid="stSidebar"]          { background-color: #ffffff; border-right: 1px solid #e9ecef; }

/* Override any inherited white text from dark themes */
html, body, [class*="st-"], .main .block-container,
p, span, li, label, div,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] *,
[data-testid="stCaptionContainer"],
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"],
.streamlit-expanderHeader,
[data-baseweb="select"] *,
[data-baseweb="input"] *,
[data-baseweb="slider"] *  { color: #2c3e50 !important; }

h1 { color: #2c3e50 !important; font-weight: 800; }
h2 { color: #34495e !important; }
h3 { color: #495057 !important; }

/* ── KPI card ── */
.kpi-card {
    background: white;
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-top: 4px solid #3498db;
    margin-bottom: 8px;
}
.kpi-card .kpi-value  { font-size: 2.2rem; font-weight: 800; color: #2c3e50; line-height: 1.1; }
.kpi-card .kpi-label  { font-size: 0.82rem; color: #7f8c8d; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-card .kpi-icon   { font-size: 1.6rem; margin-bottom: 6px; }

/* ── Elements that MUST keep white text ── */
.risk-badge, .risk-badge *,
.section-header, .section-header * { color: white !important; }

/* ── Risk badge ── */
.risk-badge {
    border-radius: 12px;
    padding: 18px 24px;
    text-align: center;
    font-size: 2rem;
    font-weight: 800;
    color: white !important;
    margin: 12px 0;
}

/* ── Insight card ── */
.insight-card {
    background: white;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    margin-bottom: 10px;
    border-left: 5px solid #3498db;
}
.insight-card .insight-icon  { font-size: 1.4rem; }
.insight-card .insight-text  { font-size: 0.95rem; color: #34495e; margin-left: 8px; }

/* ── Section header ── */
.section-header {
    background: linear-gradient(90deg, #3498db, #2980b9);
    color: white;
    padding: 10px 18px;
    border-radius: 8px;
    font-size: 1.05rem;
    font-weight: 700;
    margin: 18px 0 10px 0;
}

/* ── Explanation box ── */
.explain-box {
    background: #eaf4fb;
    border-left: 4px solid #3498db;
    border-radius: 6px;
    padding: 10px 16px;
    font-size: 0.88rem;
    color: #2c3e50;
    margin-top: -4px;
    margin-bottom: 12px;
}

/* ── Metric row ── */
.metric-row {
    display: flex;
    gap: 10px;
    margin-bottom: 14px;
}

/* ── Progress bar ── */
.prog-bar-wrap { margin: 4px 0 10px 0; }
.prog-bar-wrap .bar-label { font-size: 0.82rem; color: #555; margin-bottom: 2px; }
.prog-bar-outer { background: #e9ecef; border-radius: 20px; height: 18px; overflow: hidden; }
.prog-bar-inner { height: 18px; border-radius: 20px; transition: width 0.4s; }
.prog-bar-value  { font-size: 0.82rem; color: #555; text-align: right; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached data / model loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="📂 Loading dataset …")
def load_data() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


@st.cache_resource(show_spinner="🤖 Loading model …")
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
    return all((ART_DIR / f).exists()
               for f in ["model.joblib", "label_encoder.joblib",
                         "metrics.json", "feature_meta.json"])


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(icon: str, value: str, label: str, border_color: str = "#3498db") -> str:
    return f"""
    <div class="kpi-card" style="border-top-color:{border_color}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>"""


def risk_badge(label: str) -> str:
    color = CLASS_COLORS.get(label, "#7f8c8d")
    emoji = CLASS_EMOJI.get(label, "")
    return f"""
    <div class="risk-badge" style="background:{color}">
        {emoji} {label} Burnout Risk
    </div>"""


def insight_card(icon: str, text: str, color: str = "#3498db") -> str:
    return f"""
    <div class="insight-card" style="border-left-color:{color}">
        <span class="insight-icon">{icon}</span>
        <span class="insight-text">{text}</span>
    </div>"""


def section_header(text: str) -> None:
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def explain(text: str) -> None:
    st.markdown(f'<div class="explain-box">💡 {text}</div>', unsafe_allow_html=True)


def color_bar(value: float, max_val: float, color: str, label: str) -> str:
    pct = min(100, value / max_val * 100) if max_val else 0
    return f"""
    <div class="prog-bar-wrap">
        <div class="bar-label">{label}</div>
        <div class="prog-bar-outer">
            <div class="prog-bar-inner" style="width:{pct:.0f}%;background:{color}"></div>
        </div>
        <div class="prog-bar-value">{value:.1f} / {max_val:.0f}</div>
    </div>"""


def risk_gauge(score: float, label: str) -> go.Figure:
    """Semi-circular gauge showing burnout risk from 0 (Low) to 100 (Severe)."""
    color = CLASS_COLORS.get(label, "#7f8c8d")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(score),
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        title={"text": f"<b>Burnout Risk Score</b><br><span style='color:{color}'>{label}</span>",
               "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "#aaa", "tickvals": [0, 25, 50, 75, 100],
                     "ticktext": ["Low", "25", "Moderate", "75", "Severe"]},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#ddd",
            "steps": [
                {"range": [0,  25], "color": "#eafaf1"},
                {"range": [25, 50], "color": "#fef9e7"},
                {"range": [50, 75], "color": "#fef5e7"},
                {"range": [75, 100], "color": "#fdf2f0"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=280, margin=dict(t=60, b=10, l=20, r=20),
        paper_bgcolor="white", font={"color": "#2c3e50"},
    )
    return fig


def group_features(numeric_names, categorical_names):
    groups = {"👤 Demographics": [], "💼 Work Context": [],
              "🏃 Lifestyle": [], "🧘 Wellbeing Scores": [], "🛠 Support & Tools": []}
    demographics = {"age", "gender", "country"}
    work         = {"job_role","seniority_level","years_experience","years_at_company",
                    "company_size","industry","work_mode","salary_usd",
                    "work_hours_per_week","meetings_per_day","team_size"}
    lifestyle    = {"sleep_hours_per_night","exercise_days_per_week","vacation_days_taken"}
    support      = {"therapy_access","uses_therapy","ai_tools_daily"}
    for name in numeric_names + categorical_names:
        if name in demographics:
            groups["👤 Demographics"].append(name)
        elif name in work:
            groups["💼 Work Context"].append(name)
        elif name in lifestyle:
            groups["🏃 Lifestyle"].append(name)
        elif name in support:
            groups["🛠 Support & Tools"].append(name)
        else:
            groups["🧘 Wellbeing Scores"].append(name)
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Home Dashboard
# ─────────────────────────────────────────────────────────────────────────────
def page_home(df: pd.DataFrame, metrics: dict) -> None:
    st.markdown("# 🧠 Mental Health & Burnout Dashboard")
    st.markdown(
        "A data-driven look at burnout risk across **100,000 tech workers**. "
        "Use the sidebar to navigate between pages."
    )
    st.divider()

    # ── KPI row ───────────────────────────────────────────────────────────────
    total      = len(df)
    severe_pct = (df["burnout_level"] == "Severe").mean() * 100
    high_plus  = df["burnout_level"].isin(["High", "Severe"]).mean() * 100
    avg_hours  = df["work_hours_per_week"].mean()
    avg_sleep  = df["sleep_hours_per_night"].mean()
    model_acc  = metrics["accuracy"] * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi_card("👥", f"{total:,}", "Employees Surveyed", "#3498db"), unsafe_allow_html=True)
    c2.markdown(kpi_card("🔴", f"{severe_pct:.1f}%", "Severe Burnout", "#e74c3c"), unsafe_allow_html=True)
    c3.markdown(kpi_card("⚠️", f"{high_plus:.1f}%", "High or Severe Risk", "#e67e22"), unsafe_allow_html=True)
    c4.markdown(kpi_card("🕐", f"{avg_hours:.1f}h", "Avg Work Hours / Week", "#9b59b6"), unsafe_allow_html=True)
    c5.markdown(kpi_card("🤖", f"{model_acc:.1f}%", "Model Accuracy", "#27ae60"), unsafe_allow_html=True)

    st.markdown("")

    # ── Two-column layout: donut + insights ───────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        section_header("📊 How Is Burnout Distributed?")
        explain("Each slice shows what percentage of employees fall into that risk level.")
        counts = df["burnout_level"].value_counts().reindex(CLASS_ORDER).reset_index()
        counts.columns = ["level", "count"]
        fig = px.pie(
            counts, names="level", values="count",
            color="level", color_discrete_map=CLASS_COLORS,
            hole=0.55,
        )
        fig.update_traces(textinfo="percent+label", textfont_size=13,
                          pull=[0.04 if l == "Severe" else 0 for l in CLASS_ORDER])
        fig.update_layout(
            showlegend=True, legend=dict(orientation="h", y=-0.12),
            margin=dict(t=10, b=30, l=10, r=10), height=340,
        )
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with col_right:
        section_header("💡 Key Insights")
        explain("Automatically generated from the dataset — no technical knowledge needed.")

        top_role = df.groupby("job_role")["burnout_level"].apply(
            lambda x: (x == "Severe").mean()).idxmax()
        top_role_pct = (df[df["job_role"] == top_role]["burnout_level"] == "Severe").mean() * 100

        avg_stress_severe = df[df["burnout_level"] == "Severe"]["stress_score"].mean()
        avg_stress_low    = df[df["burnout_level"] == "Low"]["stress_score"].mean()

        overworked_severe = (df[df["burnout_level"] == "Severe"]["work_hours_per_week"] > 50).mean() * 100
        overworked_low    = (df[df["burnout_level"] == "Low"]["work_hours_per_week"] > 50).mean() * 100

        sleep_severe = df[df["burnout_level"] == "Severe"]["sleep_hours_per_night"].mean()
        sleep_low    = df[df["burnout_level"] == "Low"]["sleep_hours_per_night"].mean()

        st.markdown(insight_card(
            "🏆", f"<b>{top_role}</b> has the highest severe burnout rate at "
                  f"<b>{top_role_pct:.0f}%</b>", "#e74c3c"), unsafe_allow_html=True)
        st.markdown(insight_card(
            "😰", f"Severely burned-out workers report a stress score of "
                  f"<b>{avg_stress_severe:.1f}/10</b> vs "
                  f"<b>{avg_stress_low:.1f}/10</b> for low-risk employees", "#e67e22"),
            unsafe_allow_html=True)
        st.markdown(insight_card(
            "🕐", f"<b>{overworked_severe:.0f}%</b> of severe-burnout employees work "
                  f"50+ hours/week, vs <b>{overworked_low:.0f}%</b> of low-risk ones", "#9b59b6"),
            unsafe_allow_html=True)
        st.markdown(insight_card(
            "😴", f"Low-risk employees sleep <b>{sleep_low:.1f} hrs/night</b> on average — "
                  f"<b>{sleep_low - sleep_severe:.1f} hrs more</b> than severe-burnout workers",
            "#27ae60"), unsafe_allow_html=True)

    # ── Burnout by work mode ───────────────────────────────────────────────────
    st.divider()
    section_header("🖥️ Burnout Risk by Work Mode")
    explain("Remote, hybrid, and on-site workers show different burnout patterns.")
    grp = (df.groupby(["work_mode", "burnout_level"]).size()
           .reset_index(name="count"))
    totals = grp.groupby("work_mode")["count"].transform("sum")
    grp["share"] = grp["count"] / totals
    fig = px.bar(
        grp, x="work_mode", y="share", color="burnout_level",
        color_discrete_map=CLASS_COLORS,
        category_orders={"burnout_level": CLASS_ORDER},
        labels={"share": "Proportion", "work_mode": "Work Mode", "burnout_level": "Risk Level"},
        barmode="stack", text_auto=".0%",
    )
    fig.update_layout(yaxis_tickformat=".0%", height=380, legend_title="Risk Level",
                      margin=dict(t=20))
    st.plotly_chart(apply_theme(fig), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — Explore the Data
# ─────────────────────────────────────────────────────────────────────────────
def page_explore(df: pd.DataFrame) -> None:
    st.markdown("# 🔍 Explore the Data")
    st.caption("Use the filters on the left to focus on any group of employees.")

    # Sidebar filters
    with st.sidebar:
        st.header("🔧 Filters")
        sel_countries = st.multiselect("Country", sorted(df["country"].unique()),
                                        default=sorted(df["country"].unique()))
        sel_roles     = st.multiselect("Job role", sorted(df["job_role"].unique()),
                                        default=sorted(df["job_role"].unique()))
        sel_modes     = st.multiselect("Work mode", sorted(df["work_mode"].unique()),
                                        default=sorted(df["work_mode"].unique()))
        age_range     = st.slider("Age range",
                                   int(df["age"].min()), int(df["age"].max()),
                                   (int(df["age"].min()), int(df["age"].max())))
        hours_range   = st.slider("Work hours/week",
                                   int(df["work_hours_per_week"].min()),
                                   int(df["work_hours_per_week"].max()),
                                   (int(df["work_hours_per_week"].min()),
                                    int(df["work_hours_per_week"].max())))

    filtered = df[
        df["country"].isin(sel_countries) &
        df["job_role"].isin(sel_roles) &
        df["work_mode"].isin(sel_modes) &
        df["age"].between(*age_range) &
        df["work_hours_per_week"].between(*hours_range)
    ]
    st.info(f"Showing **{len(filtered):,}** employees matching your filters.")

    # ── Row 1: distribution + group breakdown ─────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        section_header("📊 Burnout Level Distribution")
        explain("How many employees fall into each risk category?")
        vc = filtered["burnout_level"].value_counts().reindex(CLASS_ORDER).reset_index()
        vc.columns = ["level", "count"]
        fig = px.bar(vc, x="level", y="count", color="level",
                     color_discrete_map=CLASS_COLORS,
                     category_orders={"level": CLASS_ORDER},
                     labels={"level": "Risk Level", "count": "Number of Employees"},
                     text_auto=True)
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=360, margin=dict(t=10))
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with c2:
        section_header("👥 Burnout by Group")
        explain("Compare burnout rates across different categories.")
        group_options = ["job_role", "country", "industry", "seniority_level",
                         "gender", "company_size", "work_mode"]
        col_a, col_b = st.columns([3, 1])
        group_col = col_a.selectbox("Group by", group_options, index=0)
        normalize = col_b.checkbox("Show %", value=True)
        grp = filtered.groupby([group_col, "burnout_level"]).size().reset_index(name="count")
        if normalize:
            totals = grp.groupby(group_col)["count"].transform("sum")
            grp["value"] = grp["count"] / totals
            y_col, fmt = "value", ".0%"
        else:
            grp["value"] = grp["count"]
            y_col, fmt = "value", ""
        fig = px.bar(grp, x=group_col, y=y_col, color="burnout_level",
                     color_discrete_map=CLASS_COLORS,
                     category_orders={"burnout_level": CLASS_ORDER},
                     barmode="stack",
                     labels={y_col: "Share" if normalize else "Count"})
        fig.update_layout(yaxis_tickformat=fmt, xaxis_tickangle=-30, height=360,
                          legend_title="Risk Level", margin=dict(t=10))
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    # ── Row 2: feature distribution + box plot ────────────────────────────────
    c1, c2 = st.columns(2)
    numeric_cols = [c for c in filtered.select_dtypes(include=np.number).columns
                    if c != "employee_id"]
    with c1:
        section_header("📈 Feature Distribution")
        explain("See how any numeric metric is spread across employees.")
        col_a, col_b = st.columns([3, 1])
        num_col = col_a.selectbox("Feature", numeric_cols,
                                   index=numeric_cols.index("stress_score"))
        split   = col_b.checkbox("By risk level", value=True)
        fig = px.histogram(
            filtered, x=num_col, nbins=40, opacity=0.7,
            color="burnout_level" if split else None,
            color_discrete_map=CLASS_COLORS,
            category_orders={"burnout_level": CLASS_ORDER},
            barmode="overlay" if split else "relative",
            labels={num_col: num_col.replace("_", " ").title()},
        )
        fig.update_layout(height=340, legend_title="Risk Level", margin=dict(t=10))
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with c2:
        section_header("📦 Distribution Spread by Risk Level")
        explain("The box shows the middle 50% of values; dots are outliers.")
        box_col = st.selectbox(
            "Feature for box plot", numeric_cols,
            index=numeric_cols.index("work_hours_per_week"), key="box_sel")
        fig = px.box(filtered, x="burnout_level", y=box_col,
                     color="burnout_level", color_discrete_map=CLASS_COLORS,
                     category_orders={"burnout_level": CLASS_ORDER},
                     labels={"burnout_level": "Risk Level",
                             box_col: box_col.replace("_", " ").title()})
        fig.update_layout(showlegend=False, height=340, margin=dict(t=10))
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    section_header("🔗 How Are Factors Related to Each Other?")
    explain("Positive (blue) = tend to increase together. Negative (red) = when one goes up, the other goes down.")
    key_cols = ["age", "work_hours_per_week", "sleep_hours_per_night",
                "exercise_days_per_week", "vacation_days_taken", "meetings_per_day",
                "manager_support_score", "work_life_balance_score",
                "job_satisfaction_score", "deadline_pressure_score",
                "autonomy_score", "stress_score"]
    corr = filtered[key_cols].corr().round(2)
    readable = {c: c.replace("_score", "").replace("_", " ").title() for c in key_cols}
    corr = corr.rename(index=readable, columns=readable)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=500, margin=dict(t=10, b=10))
    st.plotly_chart(apply_theme(fig), use_container_width=True)

    # ── Data table ────────────────────────────────────────────────────────────
    with st.expander("📋 View raw data table"):
        st.caption(f"Showing first 200 of {len(filtered):,} filtered rows.")
        st.dataframe(filtered.head(200), use_container_width=True)
        st.download_button(
            "⬇️ Download filtered data (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_burnout.csv", mime="text/csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — Model Performance
# ─────────────────────────────────────────────────────────────────────────────
def page_model(metrics: dict) -> None:
    st.markdown("# 📊 How Well Does Our Model Work?")
    st.caption(
        "The model was trained with hyperparameter tuning and cross-validation "
        "to ensure reliability — not just a one-off lucky result."
    )

    # ── Overall scores ────────────────────────────────────────────────────────
    section_header("🏆 Overall Performance")
    acc      = metrics["accuracy"] * 100
    macro_f1 = metrics["macro_f1"] * 100
    cv_mean  = metrics.get("cv_mean", 0) * 100
    cv_std   = metrics.get("cv_std",  0) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("🎯", f"{acc:.1f}%",      "Overall Accuracy",     "#3498db"), unsafe_allow_html=True)
    c2.markdown(kpi_card("⚖️",  f"{macro_f1:.1f}%", "Balanced Accuracy*",   "#27ae60"), unsafe_allow_html=True)
    c3.markdown(kpi_card("🔁", f"{cv_mean:.1f}%",  "CV Consistency Score", "#9b59b6"), unsafe_allow_html=True)
    c4.markdown(kpi_card("📏", f"±{cv_std:.1f}%",  "Score Variation",      "#e67e22"), unsafe_allow_html=True)
    st.caption(
        "\\* *Balanced Accuracy* treats all four risk levels equally — even the rarer ones. "
        "*CV Consistency Score* is the average across 5 independent test runs, "
        "showing the model performs reliably on unseen data."
    )

    # ── Cross-validation stability ─────────────────────────────────────────────
    if "cv_fold_scores" in metrics:
        section_header("🔁 Is the Model Consistent? (Cross-Validation)")
        explain(
            "We tested the model 5 times on different slices of data. "
            "Bars close together mean the model is reliable, not just lucky."
        )
        fold_scores = [s * 100 for s in metrics["cv_fold_scores"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Test {i+1}" for i in range(len(fold_scores))],
            y=fold_scores,
            marker_color=["#3498db" if s >= cv_mean else "#e74c3c" for s in fold_scores],
            text=[f"{s:.1f}%" for s in fold_scores],
            textposition="outside",
        ))
        fig.add_hline(y=cv_mean, line_dash="dash", line_color="#2c3e50",
                      annotation_text=f"Average: {cv_mean:.1f}%")
        fig.update_layout(
            yaxis=dict(range=[max(0, min(fold_scores) - 5), min(100, max(fold_scores) + 5)],
                       title="Balanced Accuracy (%)"),
            xaxis_title="", height=340, margin=dict(t=30),
            showlegend=False,
        )
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    # ── Model comparison ──────────────────────────────────────────────────────
    if "leaderboard" in metrics:
        section_header("🏅 Which Model Performed Best?")
        explain(
            "We trained three different AI models and a combined 'ensemble'. "
            "The winner was selected automatically based on balanced accuracy."
        )
        board = pd.DataFrame(metrics["leaderboard"]).T.reset_index()
        board.columns = ["Model", "Balanced Accuracy", "Overall Accuracy"]
        board["Balanced Accuracy"] *= 100
        board["Overall Accuracy"]  *= 100
        board = board.sort_values("Balanced Accuracy", ascending=False)
        fig = px.bar(
            board.melt(id_vars="Model", var_name="Metric", value_name="Score (%)"),
            x="Model", y="Score (%)", color="Metric", barmode="group",
            color_discrete_sequence=["#3498db", "#27ae60"],
            text_auto=".1f",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_range=[50, 80], height=380,
                          xaxis_tickangle=-15, margin=dict(t=30))
        fig.add_hline(y=70, line_dash="dot", line_color="#aaa",
                      annotation_text="70% target")
        st.plotly_chart(apply_theme(fig), use_container_width=True)
        best_model_name = metrics.get("winning_model", "")
        st.success(f"✅ **Winner: {best_model_name}** — used for all predictions")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    section_header("🗂️ Where Does the Model Get Confused?")
    explain(
        "Each row is the real risk level; each column is the model's prediction. "
        "The diagonal (dark squares) = correct. Off-diagonal = mistakes. "
        "Most mistakes happen between neighbouring levels (e.g. Moderate ↔ High)."
    )
    labels = metrics["class_labels"]
    cm     = np.array(metrics["confusion_matrix"])
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig = px.imshow(
        cm_pct, x=labels, y=labels,
        text_auto=".0f",
        color_continuous_scale="Blues",
        labels={"x": "Predicted Risk Level", "y": "Actual Risk Level", "color": "%"},
        zmin=0, zmax=100,
    )
    fig.update_layout(height=420, margin=dict(t=10))
    for i in range(len(labels)):
        fig.add_annotation(
            x=i, y=i, text="✓", showarrow=False,
            font=dict(size=18, color="white"), xref="x", yref="y",
        )
    st.plotly_chart(apply_theme(fig), use_container_width=True)

    # ── Per-class F1 ──────────────────────────────────────────────────────────
    section_header("🎯 Accuracy per Risk Level")
    explain(
        "F1 Score = how well the model identifies each specific risk level. "
        "🟢 ≥ 0.75 = Great   🟡 0.55–0.74 = Good   🔴 < 0.55 = Needs improvement."
    )
    rep  = metrics["classification_report"]
    rows = []
    for label in labels:
        r = rep[label]
        rows.append({
            "Risk Level": f"{CLASS_EMOJI.get(label,'')} {label}",
            "Precision": round(r["precision"], 3),
            "Recall":    round(r["recall"],    3),
            "F1 Score":  round(r["f1-score"],  3),
            "Count":     int(r["support"]),
        })
    class_df = pd.DataFrame(rows)
    fig = px.bar(
        class_df, x="Risk Level", y="F1 Score",
        color="F1 Score",
        color_continuous_scale=[[0, "#e74c3c"], [0.55, "#f39c12"], [0.75, "#27ae60"], [1, "#1a8a4a"]],
        range_color=[0.4, 0.95],
        text="F1 Score",
        labels={"F1 Score": "F1 Score (0 = no accuracy, 1 = perfect)"},
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.add_hline(y=0.75, line_dash="dot", line_color="#27ae60",
                  annotation_text="Great threshold (0.75)")
    fig.update_layout(showlegend=False, height=380, margin=dict(t=30),
                      yaxis_range=[0, 1.0])
    st.plotly_chart(apply_theme(fig), use_container_width=True)
    st.dataframe(class_df, use_container_width=True, hide_index=True)

    # ── Feature importance ────────────────────────────────────────────────────
    section_header("🔑 What Factors Drive Burnout the Most?")
    explain(
        "Each bar shows how much the model's accuracy drops when that factor is "
        "hidden from the model. Longer bar = more important. "
        "🔴 Orange bars = engineered features created from raw data."
    )
    imp    = pd.DataFrame(metrics["permutation_importance"][:15])
    imp["label"] = imp["feature"].str.replace("_", " ").str.title()
    engineered = {
        "stress_x_hours", "stress_x_sleep_inv", "pressure_x_imbalance",
        "stress_x_no_support", "work_recovery_ratio", "resilience_index",
        "pressure_vs_support", "seniority_ord", "company_size_ord",
        "log_salary", "log_team_size", "overworked", "sleep_deprived",
        "no_vacation", "high_meetings", "no_exercise", "meetings_burden",
        "tenure_experience_ratio", "salary_per_experience",
    }
    imp["type"] = imp["feature"].apply(lambda f: "Engineered" if f in engineered else "Original")
    fig = px.bar(
        imp.sort_values("importance"),
        x="importance", y="label", orientation="h",
        color="type", error_x="std",
        color_discrete_map={"Original": "#3498db", "Engineered": "#e67e22"},
        labels={"importance": "Impact on Model Accuracy", "label": "",
                "type": "Feature Type"},
    )
    fig.update_layout(height=520, legend=dict(orientation="h", y=1.05),
                      margin=dict(t=40, l=10))
    st.plotly_chart(apply_theme(fig), use_container_width=True)

    # ── Best hyperparameters ──────────────────────────────────────────────────
    if "best_params" in metrics:
        with st.expander("⚙️ Best Hyperparameters Found (Technical Details)"):
            st.markdown("These settings were automatically found to give the best accuracy:")
            for model_name, params in metrics["best_params"].items():
                st.markdown(f"**{model_name}**")
                st.json(params)


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 — Predict Burnout Risk
# ─────────────────────────────────────────────────────────────────────────────
def page_predict(model, label_encoder, feature_meta: dict, df: pd.DataFrame) -> None:
    st.markdown("# 🎯 Predict Burnout Risk")
    st.markdown(
        "Fill in the details below to estimate an employee's burnout risk level. "
        "All fields are pre-filled with typical values — just adjust what you know."
    )

    numeric_names     = list(feature_meta["numeric"].keys())
    categorical_names = list(feature_meta["categorical"].keys())
    iqr_bounds        = feature_meta.get("iqr_bounds", {})
    groups            = group_features(numeric_names, categorical_names)
    inputs: dict      = {}

    # Friendly display names
    FRIENDLY = {
        "age": "Age (years)", "salary_usd": "Annual Salary (USD)",
        "work_hours_per_week": "Work Hours per Week",
        "meetings_per_day": "Meetings per Day",
        "team_size": "Team Size", "sleep_hours_per_night": "Sleep Hours per Night",
        "exercise_days_per_week": "Exercise Days per Week",
        "vacation_days_taken": "Vacation Days Taken (this year)",
        "years_experience": "Years of Experience",
        "years_at_company": "Years at Current Company",
        "therapy_access": "Has Access to Therapy? (1=Yes)",
        "uses_therapy": "Currently Uses Therapy? (1=Yes)",
        "ai_tools_daily": "Uses AI Tools Daily? (1=Yes)",
        "manager_support_score": "Manager Support (1–10)",
        "work_life_balance_score": "Work-Life Balance (1–10)",
        "job_satisfaction_score": "Job Satisfaction (1–10)",
        "social_support_score": "Social Support (1–10)",
        "deadline_pressure_score": "Deadline Pressure (1–10)",
        "autonomy_score": "Autonomy / Independence (1–10)",
        "stress_score": "Stress Level (1–10)",
    }

    with st.form("predict_form"):
        for group_name, members in groups.items():
            if not members:
                continue
            with st.expander(group_name, expanded=("Work" in group_name or "Wellbeing" in group_name)):
                cols = st.columns(2)
                for i, name in enumerate(members):
                    col = cols[i % 2]
                    label = FRIENDLY.get(name, name.replace("_", " ").title())
                    if name in feature_meta["numeric"]:
                        m = feature_meta["numeric"][name]
                        lo, hi, med = m["min"], m["max"], m["median"]
                        if m["is_integer"]:
                            inputs[name] = col.slider(label, int(lo), int(hi), int(med), step=1)
                        else:
                            inputs[name] = col.slider(label, float(lo), float(hi), float(med), step=0.1)
                    else:
                        opts = feature_meta["categorical"][name]
                        inputs[name] = col.selectbox(label, opts, index=0)

        submitted = st.form_submit_button("🔍 Predict Burnout Risk", type="primary",
                                           use_container_width=True)

    if not submitted:
        st.info("👆 Adjust the sliders and dropdowns above, then click **Predict** to see the result.")
        return

    # ── Run prediction ────────────────────────────────────────────────────────
    raw_row       = pd.DataFrame([inputs])
    engineered    = engineer_features(raw_row)
    feature_order = feature_meta["feature_order"]
    X_row         = engineered[feature_order]
    proba         = model.predict_proba(X_row)[0]
    pred_idx      = int(np.argmax(proba))
    pred_label    = label_encoder.classes_[pred_idx]

    # Burnout risk score: weighted average (Low=12.5, Mod=37.5, High=62.5, Severe=87.5)
    weights    = {"Low": 12.5, "Moderate": 37.5, "High": 62.5, "Severe": 87.5}
    risk_score = sum(proba[i] * weights[label_encoder.classes_[i]]
                     for i in range(len(label_encoder.classes_)))

    st.divider()
    st.markdown("## 📋 Prediction Result")

    col_gauge, col_detail = st.columns([1, 1])

    with col_gauge:
        st.plotly_chart(risk_gauge(risk_score, pred_label), use_container_width=True)
        st.markdown(risk_badge(pred_label), unsafe_allow_html=True)

    with col_detail:
        st.markdown("### 📊 Probability Breakdown")
        explain("How confident is the model in each risk level?")
        proba_df = pd.DataFrame({
            "Risk Level": label_encoder.classes_,
            "Probability": proba,
        })
        proba_df["Risk Level"] = pd.Categorical(
            proba_df["Risk Level"], categories=CLASS_ORDER, ordered=True)
        proba_df = proba_df.sort_values("Risk Level")
        fig = px.bar(
            proba_df, x="Risk Level", y="Probability",
            color="Risk Level", color_discrete_map=CLASS_COLORS,
            text=proba_df["Probability"].apply(lambda v: f"{v:.0%}"),
        )
        fig.update_traces(textposition="outside", textfont_size=13)
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%",
                          yaxis_range=[0, 1.05], height=310, margin=dict(t=10))
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    # ── Key drivers ───────────────────────────────────────────────────────────
    st.markdown("### 🔑 What's Driving This Prediction?")
    explain("These inputs are closest to the typical profile for this risk level, "
            "based on median values across all employees.")
    medians = df.groupby("burnout_level")[[
        "stress_score", "work_hours_per_week", "sleep_hours_per_night",
        "job_satisfaction_score", "work_life_balance_score",
        "deadline_pressure_score", "vacation_days_taken",
    ]].median()

    driver_cols = st.columns(2)
    col_idx = 0
    if pred_label in medians.index:
        for col_name in medians.columns:
            val = inputs.get(col_name)
            if val is None:
                continue
            class_med  = medians.loc[pred_label, col_name]
            overall_med = df[col_name].median()
            diff = val - class_med
            rng  = df[col_name].max() - df[col_name].min()
            if rng and abs(diff) < 0.20 * rng:
                label_friendly = FRIENDLY.get(col_name, col_name.replace("_", " ").title())
                msg = (f"**{label_friendly}**: your value ({val:.1f}) "
                       f"is close to the typical {pred_label} burnout median "
                       f"({class_med:.1f})")
                driver_cols[col_idx % 2].info(msg)
                col_idx += 1

    if col_idx == 0:
        st.markdown(insight_card(
            "📌",
            "The key factors for burnout risk are: Stress Level, Work Hours, "
            "Work-Life Balance Score, Sleep Hours, and Deadline Pressure. "
            "Try adjusting these sliders to see how the prediction changes.",
            "#3498db",
        ), unsafe_allow_html=True)

    # ── Out-of-range warning ───────────────────────────────────────────────────
    warnings_list = []
    for col, b in iqr_bounds.items():
        if col in inputs and (inputs[col] < b["low"] or inputs[col] > b["high"]):
            warnings_list.append(
                f"- **{col.replace('_',' ').title()}** = {inputs[col]} "
                f"(typical range: {b['low']:.1f} – {b['high']:.1f})"
            )
    if warnings_list:
        with st.expander("⚠️ Some inputs are outside the typical training range"):
            st.markdown(
                "The model still produces a prediction, but accuracy may be lower "
                "for very unusual values.\n\n" + "\n".join(warnings_list)
            )


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # Top banner image (shown on every page)
    st.image("Brain_Burnout_Image.jpg", use_container_width=True)

    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🧠 Burnout Dashboard")
        st.markdown("---")
        page = st.radio(
            "Navigate to:",
            ["🏠 Home", "🔍 Explore Data", "📊 Model Performance", "🎯 Predict Risk"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption(
            "📦 Dataset: 100,000 tech workers\n\n"
            "🤖 Model: Soft-Voting Ensemble\n\n"
            "🔄 Tuned with RandomizedSearchCV + 5-fold CV"
        )

    if not artifacts_present():
        st.error(
            "**Model artefacts not found.** "
            "Please run `python train_model.py` first to generate the `artifacts/` folder."
        )
        st.stop()

    df      = load_data()
    metrics = load_metrics()

    if page == "🏠 Home":
        page_home(df, metrics)
    elif page == "🔍 Explore Data":
        page_explore(df)
    elif page == "📊 Model Performance":
        page_model(metrics)
    elif page == "🎯 Predict Risk":
        page_predict(load_model(), load_label_encoder(), load_feature_meta(), df)


if __name__ == "__main__":
    main()
