# dashboard/app.py
# Streamlit dashboard - Smart Production Optimization System
# Run: streamlit run dashboard/app.py
#      (run from the project root: smart_production/)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import f1_score

from data_loader import load_data, engineer_features, summary_stats, FEATURE_COLS
from optimizer   import detect_bottlenecks, suggest_workflow, oee_report
from model       import load_model, predict_live, rule_based_predict

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Production Optimizer",
    page_icon="🏭",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.title("🏭 Smart Production")
st.sidebar.markdown("---")
csv_path = st.sidebar.text_input("Sensor CSV", value="data/sensor_data.csv")

@st.cache_data
def load(path):
    return engineer_features(load_data(path))

try:
    df_full = load(csv_path)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

machines = sorted(df_full["machine_name"].unique().tolist())
selected = st.sidebar.multiselect("Filter machines", machines, default=machines)
df = df_full[df_full["machine_name"].isin(selected)].copy()

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Fault Prediction", "Workflow Optimizer", "Live Predict"],
)

st.sidebar.markdown("---")
st.sidebar.caption("C + Python | Engineering Student Project")

# ── Overview ──────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Production Overview")

    stats = summary_stats(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",     f"{len(df):,}")
    c2.metric("Total Faults",      f"{df['fault_flag'].sum():,}")
    c3.metric("Fleet Fault Rate",  f"{df['fault_flag'].mean()*100:.1f}%")
    c4.metric("Avg Output (uph)",  f"{df['output_rate_uph'].mean():.0f}")
    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        fig = px.bar(
            stats, x="machine_name", y="fault_rate_pct",
            color="fault_rate_pct", color_continuous_scale="RdYlGn_r",
            title="Fault Rate per Machine (%)",
            labels={"fault_rate_pct": "Fault %", "machine_name": ""},
        )
        fig.add_hline(y=10, line_dash="dot", line_color="orange",
                      annotation_text="10% warning level")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig2 = px.bar(
            stats, x="machine_name", y="avg_output",
            color="avg_output", color_continuous_scale="Blues",
            title="Average Output Rate (uph)",
            labels={"avg_output": "uph", "machine_name": ""},
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Time series
    df_ts = (df.groupby(["timestamp","machine_name"])["output_rate_uph"]
               .mean().reset_index())
    fig3 = px.line(
        df_ts, x="timestamp", y="output_rate_uph", color="machine_name",
        title="Output Rate Over Time",
        labels={"output_rate_uph": "Output (uph)", "timestamp": ""},
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Sensor distributions
    st.subheader("Sensor Distributions: Normal vs Fault")
    sensor = st.selectbox(
        "Sensor",
        ["temperature_c","vibration_mms","pressure_bar","current_a","output_rate_uph"],
    )
    fig4 = px.box(
        df, x="machine_name", y=sensor, color="fault_flag",
        color_discrete_map={0:"#2ecc71", 1:"#e74c3c"},
        labels={"fault_flag":"Fault (1=yes)"},
        title=f"{sensor} — Normal vs Fault",
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── Fault Prediction ──────────────────────────────────────────────────────
elif page == "Fault Prediction":
    st.title("ML Fault Prediction")

    try:
        model = load_model()
    except FileNotFoundError:
        st.warning("Model not trained. Run:  `python ml/model.py train`  from project root.")
        st.stop()

    df2 = df.copy()
    X   = df2[FEATURE_COLS].values
    df2["ml_prob"]   = model.predict_proba(X)[:, 1]
    df2["ml_pred"]   = (df2["ml_prob"] >= 0.5).astype(int)
    df2["rule_pred"] = rule_based_predict(df2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ML F1 Score",        f"{f1_score(df2['fault_flag'], df2['ml_pred']):.3f}")
    c2.metric("Rule-based F1",      f"{f1_score(df2['fault_flag'], df2['rule_pred']):.3f}")
    c3.metric("High-Risk (>70%)",   f"{(df2['ml_prob'] > 0.7).sum():,}")
    c4.metric("Missed Faults (FN)", f"{((df2['fault_flag']==1) & (df2['ml_pred']==0)).sum()}")
    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        fig = px.histogram(
            df2, x="ml_prob", color="fault_flag", nbins=40,
            color_discrete_map={0:"#2ecc71", 1:"#e74c3c"},
            title="Fault Probability Distribution",
            labels={"ml_prob":"Fault Probability","fault_flag":"Actual Fault"},
            barmode="overlay", opacity=0.75,
        )
        fig.add_vline(x=0.5, line_dash="dash", annotation_text="Threshold=0.5")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        risk = df2.groupby("machine_name").agg(
            avg_risk    = ("ml_prob", "mean"),
            high_risk_n = ("ml_prob", lambda x: (x > 0.7).sum()),
        ).reset_index()
        fig2 = px.bar(
            risk, x="machine_name", y="avg_risk",
            color="avg_risk", color_continuous_scale="RdYlGn_r",
            title="Avg ML Risk Score per Machine",
            labels={"avg_risk":"Risk Score","machine_name":""},
            range_y=[0, 1],
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Fault type pie
    code_map = {0:"None",1:"Overheating",2:"Bearing Wear",3:"Pressure Drop",4:"Electrical Surge"}
    fault_df = df2[df2["fault_flag"]==1].copy()
    fault_df["fault_type"] = fault_df["fault_code"].map(code_map)
    col_l2, col_r2 = st.columns(2)
    with col_l2:
        fig3 = px.pie(fault_df, names="fault_type", title="Fault Type Breakdown",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig3, use_container_width=True)
    with col_r2:
        fig4 = px.bar(
            fault_df.groupby(["machine_name","fault_type"]).size()
                    .reset_index(name="count"),
            x="machine_name", y="count", color="fault_type",
            title="Fault Types per Machine",
            labels={"machine_name":"","count":"Count"},
        )
        st.plotly_chart(fig4, use_container_width=True)

# ── Workflow Optimizer ────────────────────────────────────────────────────
elif page == "Workflow Optimizer":
    st.title("Workflow Optimizer")

    # OEE
    st.subheader("OEE (Overall Equipment Effectiveness)")
    oee_df = oee_report(df)
    disp   = oee_df.copy()
    for col in ["availability","performance","quality"]:
        disp[col] = (disp[col]*100).round(1).astype(str) + "%"
    disp["oee_pct"] = disp["oee_pct"].astype(str) + "%"
    st.dataframe(disp, use_container_width=True)

    fig = px.bar(
        oee_df, x="machine_name", y="oee_pct",
        color="oee_pct", color_continuous_scale="RdYlGn",
        title="OEE % per Machine", range_y=[0, 100],
        labels={"oee_pct":"OEE %","machine_name":""},
    )
    fig.add_hline(y=85, line_dash="dot", line_color="white",
                  annotation_text="World-class OEE = 85%")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Bottleneck scatter
    st.subheader("Machine Health Map")
    bn_df = detect_bottlenecks(df)
    fig2  = px.scatter(
        bn_df, x="avg_output", y="fault_rate_pct",
        color="is_bottleneck", size="total_records",
        hover_name="machine_name",
        color_discrete_map={True:"#e74c3c", False:"#2ecc71"},
        title="Output Rate vs Fault Rate  (red = bottleneck)",
        labels={"avg_output":"Avg Output (uph)","fault_rate_pct":"Fault Rate %"},
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Action recommendations
    st.subheader("Recommended Actions")
    suggestions = suggest_workflow(df)
    if not suggestions:
        st.success("No bottlenecks detected. Fleet is operating normally.")
    for s in suggestions:
        icon  = {"CRITICAL":"🔴","HIGH":"🟠","INFO":"🔵"}.get(s["priority"], "⚪")
        expanded = s["priority"] in ("CRITICAL","HIGH")
        with st.expander(f"{icon} [{s['priority']}]  {s['machine']}", expanded=expanded):
            st.markdown(f"**Reason:** {', '.join(s['reasons'])}")
            for a in s["actions"]:
                st.markdown(f"- {a}")

# ── Live Predict ──────────────────────────────────────────────────────────
elif page == "Live Predict":
    st.title("Live Sensor Prediction")
    st.markdown("Enter a current sensor reading and get an instant fault assessment.")

    try:
        load_model()
    except FileNotFoundError:
        st.error("Train the model first:  `python ml/model.py train`")
        st.stop()

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    temp   = c1.slider("Temperature (°C)",   20.0, 200.0,  75.0, 0.5)
    vib    = c1.slider("Vibration (mm/s)",    0.1,  30.0,   1.5, 0.1)
    press  = c2.slider("Pressure (bar)",      0.5,  20.0,   5.0, 0.1)
    curr   = c2.slider("Current (A)",          1.0,  60.0,  13.0, 0.5)
    cyc    = c3.slider("Cycle Time (s)",       0.5,  30.0,   3.5, 0.1)
    output = c3.slider("Output Rate (uph)",    5.0, 300.0, 100.0, 1.0)

    if st.button("Run Prediction", type="primary", use_container_width=True):
        result = predict_live({
            "temperature_c":   temp,
            "vibration_mms":   vib,
            "pressure_bar":    press,
            "current_a":       curr,
            "cycle_time_s":    cyc,
            "output_rate_uph": output,
        })

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        status_label = "⚠️ FAULT DETECTED" if result["fault_predicted"] else "✅ NORMAL"
        c1.metric("Status",       status_label)
        c2.metric("Fault Prob.",  f"{result['fault_probability']*100:.1f}%")
        c3.metric("Risk Level",   result["risk_level"])

        if result["fault_predicted"]:
            st.error(f"**Fault Type:** {result['fault_type']}")
            advice = {
                "Overheating":      ["Reduce spindle speed immediately",
                                     "Check coolant flow and heat exchangers",
                                     "Inspect thermal sensors"],
                "Bearing Wear":     ["Stop machine and inspect bearings",
                                     "Check lubrication schedule",
                                     "Verify shaft alignment"],
                "Pressure Drop":    ["Check hydraulic lines for leaks",
                                     "Inspect pump and pressure relief valves",
                                     "Replace clogged filters"],
                "Electrical Surge": ["Check supply voltage and breakers",
                                     "Inspect motor windings and connections",
                                     "Call qualified electrician"],
                "Unknown Anomaly":  ["Run full diagnostic cycle",
                                     "Compare readings against baseline",
                                     "Escalate to maintenance team"],
            }
            actions = advice.get(result["fault_type"], [])
            if actions:
                st.markdown("**Recommended actions:**")
                for a in actions:
                    st.markdown(f"- {a}")
        else:
            st.success("Machine operating within normal parameters. No action required.")

        # Gauge
        gauge_color = "#e74c3c" if result["fault_predicted"] else "#2ecc71"
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = result["fault_probability"] * 100,
            title = {"text": "Fault Risk %", "font": {"size": 18}},
            delta = {"reference": 50, "increasing": {"color": "#e74c3c"}},
            gauge = {
                "axis":  {"range": [0, 100], "tickwidth": 1},
                "bar":   {"color": gauge_color},
                "steps": [
                    {"range": [0,  40], "color": "#1a472a"},
                    {"range": [40, 70], "color": "#5d4e00"},
                    {"range": [70, 100],"color": "#6b1010"},
                ],
                "threshold": {
                    "line":  {"color": "white", "width": 3},
                    "value": 50,
                },
            },
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
