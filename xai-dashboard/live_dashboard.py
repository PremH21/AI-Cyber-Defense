"""
Live SOC Dashboard — polls the running FastAPI backend and auto-refreshes,
showing incidents as they're processed by the traffic simulator.

Requires: api server running (uvicorn) + simulator running (api/simulator.py)
Run with: streamlit run xai-dashboard/live_dashboard.py
"""
import streamlit as st
import pandas as pd
import requests
import time

st.set_page_config(page_title="Live SOC Dashboard", layout="wide")
API_URL = "http://localhost:8000"

st.title("🛡️ AI-Driven Autonomous Defense Framework — Live SOC View")
st.caption("Real-time incident stream from the running detection + response pipeline")

placeholder = st.empty()

def fetch_stats():
    try:
        return requests.get(f"{API_URL}/stats", timeout=3).json()
    except Exception:
        return None

def fetch_incidents(limit=30):
    try:
        return requests.get(f"{API_URL}/incidents", params={"limit": limit}, timeout=3).json()
    except Exception:
        return None

while True:
    with placeholder.container():
        stats = fetch_stats()
        incidents_data = fetch_incidents()

        if stats is None or incidents_data is None:
            st.error("Cannot reach API at localhost:8000 — is uvicorn running?")
            time.sleep(4)
            continue

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Processed", stats.get("total_incidents_processed", 0))
        col2.metric("Live Accuracy", f"{stats.get('accuracy', 0):.1%}" if stats.get("accuracy") else "N/A", help="No ground-truth labels exist for live-captured traffic - accuracy only applies to labeled test-set replay")
        col3.metric("Attacks Detected", stats.get("attacks_detected", 0))
        col4.metric("Avg Latency", f"{stats.get('avg_detection_latency_ms', 0):.1f}ms")

        st.divider()
        st.subheader("Recent Incidents")

        incidents = incidents_data.get("incidents", [])
        if incidents:
            df = pd.DataFrame(incidents)
            display_cols = ["timestamp", "true_label", "predicted_label", "severity",
                             "asset_criticality", "action_taken", "detection_latency_ms", "correct"]
            df_display = df[[c for c in display_cols if c in df.columns]]

            def highlight_row(row):
                if not row.get("correct", True):
                    return ["background-color: #ffcccc"] * len(row)
                elif row.get("predicted_label") == "attack":
                    return ["background-color: #fff3cd"] * len(row)
                return [""] * len(row)

            st.dataframe(df_display.style.apply(highlight_row, axis=1), width='stretch', height=500)
        else:
            st.info("No incidents yet — start the simulator: python api/simulator.py")

    time.sleep(4)
