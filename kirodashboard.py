# =============================================================================
# KIRO DASHBOARD — REAL-TIME WELL SURVEY RISK DETECTION
# -----------------------------------------------------------------------------
# Futuristic, minimal Streamlit dashboard that ingests definitive survey points
# (MD, INC, AZI, DLS) in real time and classifies each incoming station as
# RISK / NO-RISK based on Dogleg Severity (DLS).
#
# The "brain" is a pluggable model artifact (model.pkl / drilling_model.pkl).
# If a compatible classifier is present it is used; otherwise the dashboard
# gracefully falls back to a physics-based DLS-threshold rule so the app is
# always usable.
# =============================================================================

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="KIRO • Real-Time Well Risk Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# FUTURISTIC THEME (CSS)
# -----------------------------------------------------------------------------
FUTURISTIC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Rajdhani:wght@400;600&display=swap');

:root {
    --bg-0: #050915;
    --bg-1: #0a1226;
    --neon: #00e5ff;
    --neon-2: #7b5cff;
    --ok: #00ffa3;
    --warn: #ffc400;
    --danger: #ff2e63;
    --text: #dbe6ff;
}

.stApp {
    background:
        radial-gradient(1200px 600px at 15% -10%, rgba(123,92,255,0.18), transparent 60%),
        radial-gradient(1000px 500px at 100% 0%, rgba(0,229,255,0.14), transparent 55%),
        linear-gradient(180deg, var(--bg-0), var(--bg-1));
    color: var(--text);
    font-family: 'Rajdhani', sans-serif;
}

#MainMenu, footer, header {visibility: hidden;}

h1, h2, h3, .kiro-title {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 1.5px;
}

.kiro-title {
    font-size: 2.1rem;
    background: linear-gradient(90deg, var(--neon), var(--neon-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 24px rgba(0,229,255,0.25);
}
.kiro-sub {
    color: #8aa0c8;
    font-size: 0.95rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Glass cards */
.kiro-card {
    background: rgba(15, 26, 54, 0.55);
    border: 1px solid rgba(0, 229, 255, 0.18);
    border-radius: 16px;
    padding: 18px 20px;
    backdrop-filter: blur(8px);
    box-shadow: 0 0 30px rgba(0, 229, 255, 0.06), inset 0 0 20px rgba(123,92,255,0.05);
}

.metric-label { color:#7f95bd; font-size:0.8rem; letter-spacing:2px; text-transform:uppercase; }
.metric-value { font-family:'Orbitron', sans-serif; font-size:1.9rem; }
.metric-unit  { color:#7f95bd; font-size:0.9rem; }

/* Status pill */
.pill {
    display:inline-block; padding:6px 16px; border-radius:999px;
    font-family:'Orbitron', sans-serif; font-size:0.95rem; letter-spacing:2px;
}
.pill-ok     { color:var(--ok);     border:1px solid var(--ok);     box-shadow:0 0 18px rgba(0,255,163,0.35); }
.pill-warn   { color:var(--warn);   border:1px solid var(--warn);   box-shadow:0 0 18px rgba(255,196,0,0.35); }
.pill-danger { color:var(--danger); border:1px solid var(--danger); box-shadow:0 0 22px rgba(255,46,99,0.45);
               animation:pulse 1.1s infinite; }

@keyframes pulse { 0%{opacity:1;} 50%{opacity:0.55;} 100%{opacity:1;} }

.stButton>button {
    background: linear-gradient(90deg, rgba(0,229,255,0.15), rgba(123,92,255,0.15));
    border: 1px solid var(--neon);
    color: var(--text);
    border-radius: 10px;
    font-family:'Rajdhani', sans-serif; font-weight:600; letter-spacing:1px;
    transition: all .2s ease;
}
.stButton>button:hover { border-color: var(--neon-2); box-shadow:0 0 18px rgba(123,92,255,0.5); }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060b1c, #0a1330);
    border-right: 1px solid rgba(0,229,255,0.12);
}
</style>
"""
st.markdown(FUTURISTIC_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MODEL LOADER  — the "brain"
# -----------------------------------------------------------------------------
# Feature order the classifier is expected to consume. Adjust to match the
# trained model.pkl if it differs.
MODEL_FEATURES = ["measured_depth", "inclination", "azimuth", "dogleg_severity"]


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """Load a pickled classifier. Returns (model, info_str) or (None, reason)."""
    if not path or not os.path.exists(path):
        return None, f"No model file found at '{path}'."
    try:
        import joblib

        obj = joblib.load(path)
    except Exception:
        try:
            import pickle

            with open(path, "rb") as fh:
                obj = pickle.load(fh)
        except Exception as exc:  # noqa: BLE001
            return None, f"Could not load model: {exc}"

    # Model artifacts are sometimes wrapped in a dict {'model': est, ...}
    if isinstance(obj, dict):
        for key in ("model", "clf", "classifier", "estimator", "best_model"):
            if key in obj and hasattr(obj[key], "predict"):
                return obj[key], f"Loaded model['{key}'] ({type(obj[key]).__name__})."
        return None, "Pickle is a dict without a usable estimator."

    if hasattr(obj, "predict"):
        return obj, f"Loaded {type(obj).__name__}."
    return None, "Loaded object has no .predict() method."


def classify_point(model, md: float, inc: float, azi: float, dls: float,
                   dls_limit: float) -> tuple[str, float, str]:
    """
    Classify a single survey station.
    Returns (label, risk_score[0-1], source) where label is one of
    NORMAL / WARNING / CRITICAL.
    """
    # --- Try the ML brain first ------------------------------------------------
    if model is not None:
        try:
            row = {
                "measured_depth": md,
                "inclination": inc,
                "azimuth": azi,
                "dogleg_severity": dls,
            }
            X = pd.DataFrame([[row[f] for f in MODEL_FEATURES]], columns=MODEL_FEATURES)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                score = float(proba[-1])  # probability of the positive/risk class
                label = "CRITICAL" if score >= 0.5 else ("WARNING" if score >= 0.3 else "NORMAL")
                return label, score, "model"

            pred = model.predict(X)[0]
            # Regression brain (e.g. distance-to-plan): map magnitude to risk.
            if isinstance(pred, (int, float, np.floating, np.integer)) and not isinstance(pred, bool):
                val = float(pred)
                if val > 1.5 or val < -0.5:  # class-like ints -> treat >0 as risk
                    score = 1.0 if val >= 15 else min(max(val / 30.0, 0.0), 1.0)
                    label = "CRITICAL" if score >= 0.5 else ("WARNING" if score >= 0.3 else "NORMAL")
                    return label, score, "model"
                is_risk = val >= 0.5
                return ("CRITICAL" if is_risk else "NORMAL"), (1.0 if is_risk else 0.0), "model"
            # String/bool class output
            is_risk = str(pred).lower() in {"1", "true", "risk", "critical", "yes"}
            return ("CRITICAL" if is_risk else "NORMAL"), (1.0 if is_risk else 0.0), "model"
        except Exception:
            pass  # fall through to rule-based

    # --- Physics-based fallback (DLS threshold) --------------------------------
    score = min(dls / dls_limit, 1.0) if dls_limit > 0 else 0.0
    if dls > dls_limit:
        label = "CRITICAL"
    elif dls > dls_limit * 0.7:
        label = "WARNING"
    else:
        label = "NORMAL"
    return label, score, "rule"


# -----------------------------------------------------------------------------
# DOGLEG SEVERITY (minimum curvature) between two consecutive stations
# -----------------------------------------------------------------------------
def compute_dls(md0, inc0, azi0, md1, inc1, azi1) -> float:
    """Dogleg severity in deg/30m between two survey stations."""
    L = md1 - md0
    if L <= 0:
        return 0.0
    i0, i1 = np.radians(inc0), np.radians(inc1)
    a0, a1 = np.radians(azi0), np.radians(azi1)
    arg = np.sin(i0) * np.sin(i1) * np.cos(a1 - a0) + np.cos(i0) * np.cos(i1)
    arg = np.clip(arg, -1.0, 1.0)
    dogleg = np.degrees(np.arccos(arg))
    return float(dogleg * (30.0 / L))


# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------
@dataclass
class SurveyStream:
    rows: list = field(default_factory=list)   # list of dicts
    cursor: int = 0                             # feed pointer for replay
    running: bool = False


if "stream" not in st.session_state:
    st.session_state.stream = SurveyStream()
if "log" not in st.session_state:
    st.session_state.log = []


def reset_stream():
    st.session_state.stream = SurveyStream()
    st.session_state.log = []


def _synthetic_survey(n: int = 120, seed: int = 7) -> pd.DataFrame:
    """Generate a plausible definitive survey with a couple of dogleg events."""
    rng = np.random.default_rng(seed)
    md = np.arange(0, n) * 30.0
    inc = np.clip(np.cumsum(rng.normal(0.6, 0.25, n)), 0, 90)
    azi = (120 + np.cumsum(rng.normal(0.0, 0.8, n))) % 360
    # inject two hard doglegs
    for spike in (45, 90):
        inc[spike:spike + 3] += 6.0
        azi[spike:spike + 3] += 10.0
    return pd.DataFrame({"MD": md, "INC": inc, "AZI": azi})


# -----------------------------------------------------------------------------
# SIDEBAR — controls & data source
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='kiro-title' style='font-size:1.3rem'>KIRO CONTROL</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='kiro-sub'>Real-Time Survey Feed</div>", unsafe_allow_html=True)
    st.write("")

    model_path = st.text_input("Model artifact (.pkl)", value="drilling_model.pkl",
                               help="Pluggable classifier brain. Falls back to DLS rule if unavailable.")
    model, model_info = load_model(model_path)
    if model is not None:
        st.success(f"🧠 Brain online — {model_info}")
    else:
        st.warning(f"🧠 Brain offline — using DLS rule.\n\n{model_info}")

    st.divider()
    dls_limit = st.slider("Safe DLS limit (deg/30m)", 1.0, 6.0, 3.0, 0.1)
    speed = st.slider("Feed speed (sec / station)", 0.1, 2.0, 0.4, 0.1)

    st.divider()
    st.markdown("<div class='kiro-sub'>Data Source</div>", unsafe_allow_html=True)
    src = st.radio("Source", ["Synthetic demo feed", "Upload CSV/Excel"],
                   label_visibility="collapsed")

    survey_df: Optional[pd.DataFrame] = None
    if src == "Upload CSV/Excel":
        up = st.file_uploader("Definitive survey (MD, INC, AZI)", type=["csv", "xlsx"])
        if up is not None:
            survey_df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
    else:
        survey_df = _synthetic_survey()

    # Normalize column names -> MD / INC / AZI
    if survey_df is not None:
        colmap = {}
        for c in survey_df.columns:
            cu = str(c).strip().upper()
            if cu in {"MD", "MEASURED DEPTH", "DEPTH"}:
                colmap[c] = "MD"
            elif cu in {"INC", "INCLINATION", "INCL"}:
                colmap[c] = "INC"
            elif cu in {"AZI", "AZIMUTH", "AZIM"}:
                colmap[c] = "AZI"
        survey_df = survey_df.rename(columns=colmap)

    st.write("")
    c1, c2 = st.columns(2)
    start = c1.button("▶ START", use_container_width=True)
    stop = c2.button("⏸ PAUSE", use_container_width=True)
    step = st.button("⏭ STEP ONE STATION", use_container_width=True)
    if st.button("⟲ RESET", use_container_width=True):
        reset_stream()

    if start:
        st.session_state.stream.running = True
    if stop:
        st.session_state.stream.running = False

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown("<div class='kiro-title'>🛰️ REAL-TIME WELL RISK DETECTION</div>",
            unsafe_allow_html=True)
st.markdown(
    "<div class='kiro-sub'>Definitive Survey • Dogleg Severity Classification • Early Warning</div>",
    unsafe_allow_html=True,
)
st.write("")


# -----------------------------------------------------------------------------
# FEED ENGINE — ingest the next station into the live stream
# -----------------------------------------------------------------------------
def ingest_next(source_df: pd.DataFrame):
    stream = st.session_state.stream
    if source_df is None or stream.cursor >= len(source_df):
        stream.running = False
        return False

    r = source_df.iloc[stream.cursor]
    md, inc, azi = float(r["MD"]), float(r["INC"]), float(r["AZI"])

    if stream.rows:
        prev = stream.rows[-1]
        dls = compute_dls(prev["MD"], prev["INC"], prev["AZI"], md, inc, azi)
    else:
        dls = 0.0

    label, score, source = classify_point(model, md, inc, azi, dls, dls_limit)
    rec = {"MD": md, "INC": inc, "AZI": azi, "DLS": dls,
           "Status": label, "RiskScore": score, "Source": source}
    stream.rows.append(rec)
    if label == "CRITICAL":
        st.session_state.log.append(
            f"⛔ MD {md:,.0f} m — DLS {dls:.2f}°/30m → {label} (score {score:.2f}, via {source})"
        )
    stream.cursor += 1
    return True


# Drive the feed
if step and survey_df is not None:
    ingest_next(survey_df)
if st.session_state.stream.running and survey_df is not None:
    ingest_next(survey_df)


# -----------------------------------------------------------------------------
# LIVE VIEW
# -----------------------------------------------------------------------------
stream = st.session_state.stream
live = pd.DataFrame(stream.rows)

# --- Current-status banner -----------------------------------------------------
if not live.empty:
    latest = live.iloc[-1]
    pill_class = {"NORMAL": "pill-ok", "WARNING": "pill-warn", "CRITICAL": "pill-danger"}[latest["Status"]]
    banner_txt = {"NORMAL": "◉ TRAJECTORY SAFE", "WARNING": "△ APPROACHING LIMIT",
                  "CRITICAL": "⛔ DOGLEG RISK DETECTED"}[latest["Status"]]
    st.markdown(
        f"<div class='kiro-card' style='text-align:center'>"
        f"<span class='pill {pill_class}'>{banner_txt}</span></div>",
        unsafe_allow_html=True,
    )
    st.write("")

# --- KPI row -------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)


def kpi(col, label, value, unit="", color="var(--neon)"):
    col.markdown(
        f"<div class='kiro-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value' style='color:{color}'>{value}"
        f"<span class='metric-unit'> {unit}</span></div></div>",
        unsafe_allow_html=True,
    )


if live.empty:
    kpi(k1, "Current MD", "—", "m")
    kpi(k2, "Inclination", "—", "°")
    kpi(k3, "Azimuth", "—", "°")
    kpi(k4, "Dogleg Severity", "—", "°/30m")
    kpi(k5, "Risk Stations", "0")
else:
    latest = live.iloc[-1]
    dls_color = ("var(--danger)" if latest["Status"] == "CRITICAL"
                 else "var(--warn)" if latest["Status"] == "WARNING" else "var(--ok)")
    n_crit = int((live["Status"] == "CRITICAL").sum())
    kpi(k1, "Current MD", f"{latest['MD']:,.0f}", "m")
    kpi(k2, "Inclination", f"{latest['INC']:.2f}", "°")
    kpi(k3, "Azimuth", f"{latest['AZI']:.2f}", "°")
    kpi(k4, "Dogleg Severity", f"{latest['DLS']:.2f}", "°/30m", dls_color)
    kpi(k5, "Risk Stations", f"{n_crit}", "", "var(--danger)" if n_crit else "var(--ok)")

st.write("")

# --- Live DLS chart ------------------------------------------------------------
left, right = st.columns([2, 1])

with left:
    st.markdown("<div class='kiro-sub'>Dogleg Severity vs Depth</div>", unsafe_allow_html=True)
    fig = go.Figure()
    if not live.empty:
        colors = live["Status"].map(
            {"NORMAL": "#00ffa3", "WARNING": "#ffc400", "CRITICAL": "#ff2e63"}
        )
        fig.add_trace(go.Scatter(
            x=live["MD"], y=live["DLS"], mode="lines",
            line=dict(color="#00e5ff", width=2), name="DLS",
        ))
        fig.add_trace(go.Scatter(
            x=live["MD"], y=live["DLS"], mode="markers",
            marker=dict(color=colors, size=7, line=dict(color="#050915", width=1)),
            name="Station", hovertemplate="MD %{x:.0f} m<br>DLS %{y:.2f}°/30m<extra></extra>",
        ))
        fig.add_hline(y=dls_limit, line_dash="dash", line_color="#ff2e63",
                      annotation_text=f"Limit {dls_limit}°/30m",
                      annotation_font_color="#ff2e63")
    fig.update_layout(
        template="plotly_dark", height=340,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,18,38,0.4)",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Measured Depth (m)", yaxis_title="DLS (deg/30m)",
        font=dict(family="Rajdhani"),
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("<div class='kiro-sub'>Risk Gauge</div>", unsafe_allow_html=True)
    score = float(live.iloc[-1]["RiskScore"]) if not live.empty else 0.0
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={"suffix": "%", "font": {"color": "#dbe6ff"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8aa0c8"},
            "bar": {"color": "#00e5ff"},
            "bgcolor": "rgba(10,18,38,0.4)",
            "steps": [
                {"range": [0, 30], "color": "rgba(0,255,163,0.25)"},
                {"range": [30, 50], "color": "rgba(255,196,0,0.25)"},
                {"range": [50, 100], "color": "rgba(255,46,99,0.30)"},
            ],
            "threshold": {"line": {"color": "#ff2e63", "width": 3},
                          "thickness": 0.8, "value": 50},
        },
    ))
    gauge.update_layout(
        template="plotly_dark", height=340,
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="Orbitron"),
    )
    st.plotly_chart(gauge, use_container_width=True)

# --- Alert log + station table -------------------------------------------------
st.write("")
c_log, c_tbl = st.columns([1, 2])

with c_log:
    st.markdown("<div class='kiro-sub'>Alert Log</div>", unsafe_allow_html=True)
    if st.session_state.log:
        st.markdown(
            "<div class='kiro-card' style='max-height:260px; overflow:auto'>"
            + "<br>".join(f"<span style='color:#ff2e63'>{x}</span>"
                          for x in reversed(st.session_state.log[-30:]))
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div class='kiro-card' style='color:#00ffa3'>No risks detected.</div>",
                    unsafe_allow_html=True)

with c_tbl:
    st.markdown("<div class='kiro-sub'>Recent Stations</div>", unsafe_allow_html=True)
    if not live.empty:
        show = live.tail(12).iloc[::-1][["MD", "INC", "AZI", "DLS", "RiskScore", "Status", "Source"]]
        st.dataframe(
            show.style.format({"MD": "{:.0f}", "INC": "{:.2f}", "AZI": "{:.2f}",
                               "DLS": "{:.2f}", "RiskScore": "{:.2f}"}),
            use_container_width=True, height=360,
        )
    else:
        st.markdown("<div class='kiro-card'>Awaiting survey stream… press ▶ START.</div>",
                    unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# AUTO-REFRESH LOOP  (drives the real-time feed)
# -----------------------------------------------------------------------------
if st.session_state.stream.running and survey_df is not None and stream.cursor < len(survey_df):
    time.sleep(float(speed))
    st.rerun()
