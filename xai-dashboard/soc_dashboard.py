import html
import os
from datetime import datetime

import altair as alt
import pandas as pd
import requests
import streamlit as st

API = os.environ.get("SOC_API", "http://localhost:8000")
BAND_COLOR = {"CRITICAL": "#ff3b5c", "HIGH": "#ff8a3d", "MEDIUM": "#ffd23f", "LOW": "#3ddc97", "NONE": "#6c7a96"}

st.set_page_config(page_title="AI Cyber Defense - SOC", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.stApp {background: radial-gradient(1200px 600px at 10% -10%, #14213d 0%, #0a0f1e 55%, #070a14 100%); color:#e6ecff;}
header[data-testid="stHeader"] {background: transparent;}
h1,h2,h3 {letter-spacing:.3px;}
.card {background:linear-gradient(145deg,rgba(255,255,255,.07),rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.10);
  border-radius:16px;padding:16px 18px;backdrop-filter:blur(6px);box-shadow:0 8px 28px rgba(0,0,0,.35);}
.kpi-label {font-size:.78rem;text-transform:uppercase;letter-spacing:1.2px;color:#8fa2d6;}
.kpi-value {font-size:2rem;font-weight:800;line-height:1.15;background:linear-gradient(90deg,#7cf3ff,#8a7cff);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.banner {border-radius:16px;padding:18px 24px;font-size:1.25rem;font-weight:800;letter-spacing:.6px;margin-bottom:14px;}
.pulse {animation:pulse 1.2s infinite;}
@keyframes pulse {0%{box-shadow:0 0 0 0 rgba(255,59,92,.65);}70%{box-shadow:0 0 0 18px rgba(255,59,92,0);}100%{box-shadow:0 0 0 0 rgba(255,59,92,0);}}
.dot {display:inline-block;width:10px;height:10px;border-radius:50%;background:#3ddc97;margin-right:8px;animation:blink 1.4s infinite;}
@keyframes blink {0%,100%{opacity:1;}50%{opacity:.25;}}
.chip {display:inline-block;padding:3px 11px;border-radius:999px;font-size:.78rem;margin:2px 6px 2px 0;
  background:rgba(124,243,255,.12);border:1px solid rgba(124,243,255,.35);color:#7cf3ff;}
.sevbar {height:12px;border-radius:8px;background:rgba(255,255,255,.1);overflow:hidden;margin:6px 0 2px;}
.sevfill {height:100%;border-radius:8px;transition:width .6s;}
.why {border-left:4px solid #8a7cff;padding:10px 14px;background:rgba(138,124,255,.10);border-radius:8px;font-size:.95rem;}
.small {color:#8fa2d6;font-size:.82rem;}
</style>
""", unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x))


def get(path, params=None):
    try:
        r = requests.get(API + path, params=params, timeout=4)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def show_chart(c):
    c = c.configure(background="rgba(0,0,0,0)").configure_view(strokeWidth=0) \
        .configure_axis(labelColor="#b9c6ee", titleColor="#8fa2d6", gridColor="rgba(255,255,255,.07)") \
        .configure_legend(labelColor="#b9c6ee", titleColor="#8fa2d6")
    try:
        st.altair_chart(c, width="stretch")
    except TypeError:
        st.altair_chart(c, use_container_width=True)


def show_table(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


def kpi(label, value):
    return f'<div class="card"><div class="kpi-label">{esc(label)}</div><div class="kpi-value">{esc(value)}</div></div>'


def color_scale():
    return alt.Scale(domain=list(BAND_COLOR), range=list(BAND_COLOR.values()))


def keep(rows, cutoff):
    out = [r for r in (rows or []) if isinstance(r.get("timestamp"), (int, float)) and r["timestamp"] >= cutoff]
    return sorted(out, key=lambda r: r["timestamp"], reverse=True)


def addr(r):
    return f'{r.get("src_ip", "?")}:{r.get("src_port", 0)} -> {r.get("dst_ip", "?")}:{r.get("dst_port", 0)}'


def detail(r):
    band = r.get("severity_band", "NONE")
    col = BAND_COLOR.get(band, "#6c7a96")
    sev = int(r.get("severity", 0))
    st.markdown(f"""
<div class="card" style="border-color:{col};">
  <div style="font-size:1.35rem;font-weight:800;">{esc(r.get("attack_type", "?"))}</div>
  <div class="small">{esc(r.get("category", ""))} &middot; slide category: {esc(r.get("slide_category", ""))}</div>
  <div class="sevbar"><div class="sevfill" style="width:{sev * 10}%;background:linear-gradient(90deg,#3ddc97,{col});"></div></div>
  <div><b style="color:{col};">Severity {sev}/10 - {esc(band)}</b> &nbsp; confidence {float(r.get("confidence") or 0):.0%}</div>
  <div style="margin-top:8px;">Action decided: <b>{esc(r.get("action_taken", "?"))}</b>
     <span class="small">(chosen by: {esc(r.get("action_source", "?"))})</span></div>
  <div style="margin-top:6px;">Address: <code>{esc(addr(r))}</code> &nbsp; {esc(r.get("protocol", ""))} {esc(r.get("ip_version", ""))}</div>
  <div class="small">Asset criticality: {esc(r.get("asset_criticality", "?"))} &middot; incident {esc(r.get("incident_id", "-"))}
     &middot; repeats {esc(r.get("repeat_count", 1))} &middot; latency {esc(r.get("latency_ms", "?"))} ms</div>
  <div style="margin-top:8px;">{"".join(f'<span class="chip">{esc(x)}</span>' for x in r.get("detection_layers", []))}</div>
</div>""", unsafe_allow_html=True)
    st.markdown(f'<div class="why" style="margin-top:12px;">{esc(r.get("why", ""))}</div>', unsafe_allow_html=True)

    ex = r.get("explanation") or {}
    st.markdown("#### Explainability")
    if ex.get("type") == "shap":
        d = pd.DataFrame(ex["top_5_features"])
        d["label"] = d.apply(lambda x: f'{x["feature"]} = {x["value"]:g}', axis=1)
        d["effect"] = d["shap_contribution"].apply(lambda v: "pushes toward attack" if v > 0 else "pushes away")
        c = alt.Chart(d).mark_bar(cornerRadius=5).encode(
            x=alt.X("shap_contribution:Q", title="SHAP contribution (model output units)"),
            y=alt.Y("label:N", sort=None, title=None),
            color=alt.Color("effect:N", scale=alt.Scale(domain=["pushes toward attack", "pushes away"],
                                                        range=["#ff5d7a", "#4cc9f0"]), title=None),
            tooltip=["feature", "value", "shap_contribution"]).properties(height=230)
        show_chart(c)
        st.caption(f'Top-5 SHAP features for class: {ex.get("explained_class", "")}. Red bars raise the attack score, blue bars lower it.')
    elif ex.get("type") == "autoencoder_reconstruction_error":
        d = pd.DataFrame(ex["top_5_features"])
        d["label"] = d.apply(lambda x: f'{x["feature"]} = {x["value"]:g}', axis=1)
        c = alt.Chart(d).mark_bar(cornerRadius=5, color="#ffb347").encode(
            x=alt.X("reconstruction_error:Q", title="Reconstruction error (how abnormal)"),
            y=alt.Y("label:N", sort=None, title=None), tooltip=["feature", "value", "reconstruction_error"]).properties(height=230)
        show_chart(c)
        st.caption("Autoencoder explanation: features it could least reconstruct from normal-traffic patterns.")
    elif ex.get("type") == "behavioural_evidence":
        a, b, c3 = st.columns(3)
        if "observed" in ex:
            a.markdown(kpi("Observed", f'{ex["observed"]:g}'), unsafe_allow_html=True)
            b.markdown(kpi("Rule threshold", f'{ex["threshold"]:g}'), unsafe_allow_html=True)
            c3.markdown(kpi("Window (s)", ex.get("window_seconds", "-")), unsafe_allow_html=True)
        ev = ex.get("evidence") or {}
        if ev:
            show_table(pd.DataFrame([{"evidence": k, "value": str(v)} for k, v in ev.items()]))
    else:
        st.info("No explanation attached to this record.")

    u = (r.get("ml") or {}).get("unsupervised")
    if u:
        with st.expander("Model scores"):
            ml = r["ml"]
            st.write(f'Supervised top class: **{ml.get("top_class")}** (p={ml.get("top_probability")}), '
                     f'attack probability {ml.get("attack_probability")}')
            st.write(f'Isolation Forest score {u["iforest_score"]} (flag below {u["iforest_threshold"]}) | '
                     f'Autoencoder error {u["autoencoder_error"]} (flag above {u["autoencoder_threshold"]})')


def render(window):
    stats = get("/stats")
    health = get("/health") or {}
    if stats is None:
        st.markdown('<div class="banner" style="background:#3a1020;border:1px solid #ff3b5c;">Service offline - start: '
                    'uvicorn api.live_service:app --port 8000</div>', unsafe_allow_html=True)
        return
    cutoff = 0 if window == "All" else datetime.now().timestamp() - {"15 min": 900, "1 hour": 3600, "24 hours": 86400}[window]
    inc = keep((get("/incidents", {"limit": 300}) or {}).get("incidents"), cutoff)
    flows = keep((get("/flows", {"limit": 400}) or {}).get("flows"), cutoff)
    watch = [f for f in flows if f.get("verdict") == "anomaly_watch"]

    st.markdown(f'<h2><span class="dot"></span>AI-Driven Autonomous Defense - Live SOC</h2>'
                f'<div class="small">engine {"loaded" if health.get("engine_loaded") else "NOT loaded"} &middot; '
                f'RL policy states {health.get("rl_policy_states", 0)} &middot; storage {esc(health.get("storage", "?"))} '
                f'&middot; updated {datetime.now():%H:%M:%S}</div>', unsafe_allow_html=True)

    now = datetime.now().timestamp()
    recent = [r for r in inc if now - r["timestamp"] <= 300]
    top = max(recent, key=lambda r: r.get("severity", 0)) if recent else None
    if top is None:
        st.markdown('<div class="banner" style="background:rgba(61,220,151,.12);border:1px solid #3ddc97;color:#3ddc97;">'
                    'ALL CLEAR - no incidents in the last 5 minutes</div>', unsafe_allow_html=True)
    else:
        b = top.get("severity_band", "LOW")
        col = BAND_COLOR[b] if b in BAND_COLOR else "#ffd23f"
        cls = "banner pulse" if b in ("HIGH", "CRITICAL") else "banner"
        st.markdown(f'<div class="{cls}" style="background:{col}22;border:1px solid {col};color:{col};">'
                    f'ACTIVE THREAT: {esc(top.get("attack_type"))} &middot; severity {top.get("severity")}/10 {esc(b)} &middot; '
                    f'action {esc(top.get("action_taken"))}<br><span style="font-size:.9rem;font-weight:500;">{esc(addr(top))}</span></div>',
                    unsafe_allow_html=True)

    acts = [r.get("action_taken") for r in inc]
    cols = st.columns(6)
    items = [("Flows analyzed", stats.get("total_flows_analyzed", 0)), ("Incidents (window)", len(inc)),
             ("Flagged rate", f'{(stats.get("flagged_flow_rate") or 0):.1%}'),
             ("Avg latency", f'{stats.get("avg_detection_latency_ms") or 0:.1f} ms'),
             ("Block / quarantine", sum(a in ("block", "quarantine", "rollback") for a in acts)), ("Watch items", len(watch))]
    for c, (l, v) in zip(cols, items):
        c.markdown(kpi(l, v), unsafe_allow_html=True)
    st.write("")

    d = pd.DataFrame([{"time": datetime.fromtimestamp(r["timestamp"]), "severity": r.get("severity", 0),
                       "severity_band": r.get("severity_band", "NONE"), "attack_type": r.get("attack_type"),
                       "category": r.get("category"), "action": r.get("action_taken"), "target": addr(r)} for r in inc])
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Incident timeline")
        if d.empty:
            st.info("No incidents in this window. Normal traffic is being analysed and stays quiet.")
        else:
            show_chart(alt.Chart(d).mark_circle(opacity=.85).encode(
                x=alt.X("time:T", title=None), y=alt.Y("severity:Q", scale=alt.Scale(domain=[0, 10.5])),
                size=alt.Size("severity:Q", legend=None, scale=alt.Scale(range=[90, 650])),
                color=alt.Color("severity_band:N", scale=color_scale(), title="Band"),
                tooltip=["time", "attack_type", "severity", "action", "target"]).properties(height=260))
    with right:
        st.markdown("#### Traffic verdicts (recent flows)")
        if flows:
            v = pd.Series([f.get("verdict", "benign") for f in flows]).value_counts().reset_index()
            v.columns = ["verdict", "count"]
            show_chart(alt.Chart(v).mark_arc(innerRadius=60).encode(
                theta="count:Q", color=alt.Color("verdict:N", scale=alt.Scale(scheme="tableau10")),
                tooltip=["verdict", "count"]).properties(height=260))
        else:
            st.info("No flows yet.")

    if not d.empty:
        c1, c2, c3 = st.columns(3)
        for col_, field, title, scheme in ((c1, "attack_type", "Attack types", "turbo"), (c2, "category", "Categories", "set2"),
                                           (c3, "action", "Actions decided", "tealblues")):
            g = d.groupby(field).size().reset_index(name="count")
            with col_:
                st.markdown(f"#### {title}")
                show_chart(alt.Chart(g).mark_bar(cornerRadius=5).encode(
                    x=alt.X("count:Q", title=None), y=alt.Y(f"{field}:N", sort="-x", title=None),
                    color=alt.Color(f"{field}:N", scale=alt.Scale(scheme=scheme), legend=None),
                    tooltip=[field, "count"]).properties(height=200))

    t1, t2, t3, t4 = st.tabs(["Incidents", "Watch list", "Live flows", "How it decides"])
    with t1:
        if not inc:
            st.info("Nothing to show. Run a real test from a second device and incidents will appear here with full detail.")
        else:
            show_table(pd.DataFrame([{"time": f'{x["time"]:%H:%M:%S}', "severity": x["severity"], "band": x["severity_band"],
                                      "type": x["attack_type"], "category": x["category"], "action": x["action"],
                                      "source -> destination": x["target"]} for x in d.to_dict("records")]))
            labels = {f'{datetime.fromtimestamp(r["timestamp"]):%H:%M:%S} | sev {r.get("severity")} | {r.get("attack_type")} | {r.get("incident_id", "-")}': r
                      for r in inc[:60]}
            choice = st.selectbox("Inspect incident (newest first)", list(labels), key="sel")
            detail(labels[choice])
    with t2:
        st.caption("Watch = weak signal. Logged for analysts, never escalated or blocked automatically.")
        if watch:
            show_table(pd.DataFrame([{"time": f'{datetime.fromtimestamp(f["timestamp"]):%H:%M:%S}', "type": f.get("attack_type"),
                                      "layers": ", ".join(f.get("detection_layers", [])), "target": addr(f)} for f in watch[:100]]))
        else:
            st.info("No watch items.")
    with t3:
        if flows:
            show_table(pd.DataFrame([{"time": f'{datetime.fromtimestamp(f["timestamp"]):%H:%M:%S}', "verdict": f.get("verdict"),
                                      "proto": f.get("protocol"), "flow": addr(f), "severity": f.get("severity"),
                                      "latency ms": f.get("latency_ms")} for f in flows[:150]]))
    with t4:
        st.markdown("""
**Pipeline per flow:** read 16 flow features -> supervised XGBoost (known attacks) + Isolation Forest and Autoencoder
(unknown behaviour, trained on benign only) + behavioural rules (scans, brute force, floods, lateral movement, beaconing,
exfiltration, ARP spoofing) -> severity 1-10 -> RL policy picks the action -> SHAP / evidence explanation -> stored.

**Honest limits (stated, not hidden):**
- The action column is the decision the policy made and logged. This build does not change firewall rules itself.
- Web attacks (SQLi, XSS) and phishing are invisible to flow statistics; bots are only partly caught.
- Unsupervised-only findings are capped to Watch / alert-only because they are noisy on modern traffic.
- The RL policy was trained in a simulated environment, since real response outcomes do not exist yet.
""")


with st.sidebar:
    st.markdown("### Controls")
    window = st.selectbox("Time window", ["15 min", "1 hour", "24 hours", "All"], index=3)
    live = st.toggle("Live refresh (3 s)", True)
    st.caption(f"API: {API}")

st.fragment(render, run_every="3s" if live else None)(window)
