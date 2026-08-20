import json
import os

UNSW_JSON = "ml-engine/models/shap_explanations_unsw.json"
CICIDS_JSON = "ml-engine/models/shap_explanations_cicids.json"
OUT_PATH = "xai-dashboard/shap_dashboard.html"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>XAI Threat Explainability Panel — AI-Driven Autonomous Defense Framework</title>
<style>
  :root {
    --bg: #0b0f14;
    --panel: #121820;
    --panel-border: #1f2a36;
    --text: #d7e0e8;
    --text-dim: #7c8a99;
    --accent-attack: #e5484d;
    --accent-benign: #3ecf8e;
    --accent-neutral: #4a90e2;
    --mono: 'SF Mono', 'Consolas', 'Menlo', monospace;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, 'Segoe UI', sans-serif;
    padding: 24px;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    border-bottom: 1px solid var(--panel-border);
    padding-bottom: 16px;
    margin-bottom: 20px;
  }
  h1 { font-size: 18px; font-weight: 600; margin: 0; letter-spacing: 0.02em; }
  .subtitle { color: var(--text-dim); font-size: 12px; font-family: var(--mono); }
  .dataset-tabs { display: flex; gap: 8px; margin-bottom: 16px; }
  .tab-btn {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    color: var(--text-dim);
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    font-family: var(--mono);
  }
  .tab-btn.active { color: var(--text); border-color: var(--accent-neutral); background: #16202c; }
  .layout { display: grid; grid-template-columns: 320px 1fr; gap: 16px; }
  .incident-list {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    max-height: 70vh;
    overflow-y: auto;
  }
  .incident-item {
    padding: 12px 14px;
    border-bottom: 1px solid var(--panel-border);
    cursor: pointer;
    font-family: var(--mono);
    font-size: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .incident-item:hover { background: #16202c; }
  .incident-item.selected { background: #16202c; border-left: 3px solid var(--accent-neutral); }
  .badge {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.04em;
  }
  .badge.attack { background: rgba(229,72,77,0.15); color: var(--accent-attack); }
  .badge.benign { background: rgba(62,207,142,0.15); color: var(--accent-benign); }
  .detail-panel {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    padding: 24px;
  }
  .verdict-row { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }
  .verdict-label { font-size: 22px; font-weight: 700; }
  .verdict-label.attack { color: var(--accent-attack); }
  .verdict-label.benign { color: var(--accent-benign); }
  .meta { color: var(--text-dim); font-family: var(--mono); font-size: 12px; margin-bottom: 24px; }
  .section-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-dim);
    margin-bottom: 12px;
  }
  .bar-row { margin-bottom: 14px; }
  .bar-label-row { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 12px; margin-bottom: 4px; }
  .bar-feature-name { color: var(--text); }
  .bar-value { color: var(--text-dim); }
  .bar-track { background: #0b0f14; border-radius: 4px; height: 18px; position: relative; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; position: absolute; top: 0; }
  .bar-fill.pos { background: var(--accent-attack); left: 50%; }
  .bar-fill.neg { background: var(--accent-benign); right: 50%; }
  .bar-center-line { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: var(--panel-border); }
  .contribution-text { font-family: var(--mono); font-size: 11px; color: var(--text-dim); margin-top: 2px; }
  .explain-footer {
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid var(--panel-border);
    color: var(--text-dim);
    font-size: 12px;
    line-height: 1.6;
  }
</style>
</head>
<body>

<header>
  <div>
    <h1>XAI Threat Explainability Panel</h1>
    <div class="subtitle">Innovation 6 — SHAP-based per-incident decision auditing</div>
  </div>
  <div class="subtitle">AI-Driven Autonomous Defense Framework · BLDEA CET</div>
</header>

<div class="dataset-tabs">
  <button class="tab-btn active" data-dataset="unsw">UNSW-NB15</button>
  <button class="tab-btn" data-dataset="cicids">CIC-IDS-2017</button>
</div>

<div class="layout">
  <div class="incident-list" id="incidentList"></div>
  <div class="detail-panel" id="detailPanel"></div>
</div>

<script>
const DATA = __DATA_PLACEHOLDER__;

let currentDataset = "unsw";
let currentIndex = 0;

function renderList() {
  const list = document.getElementById("incidentList");
  list.innerHTML = "";
  DATA[currentDataset].forEach((item, i) => {
    const div = document.createElement("div");
    div.className = "incident-item" + (i === currentIndex ? " selected" : "");
    div.innerHTML = `
      <span>Row #${item.test_row_index}</span>
      <span class="badge ${item.true_label}">${item.true_label.toUpperCase()}</span>
    `;
    div.onclick = () => { currentIndex = i; renderList(); renderDetail(); };
    list.appendChild(div);
  });
}

function renderDetail() {
  const item = DATA[currentDataset][currentIndex];
  const panel = document.getElementById("detailPanel");
  const maxAbs = Math.max(...item.top_5_contributing_features.map(f => Math.abs(f.shap_contribution)));

  let barsHtml = "";
  item.top_5_contributing_features.forEach(f => {
    const pct = (Math.abs(f.shap_contribution) / maxAbs) * 48;
    const isPos = f.shap_contribution > 0;
    barsHtml += `
      <div class="bar-row">
        <div class="bar-label-row">
          <span class="bar-feature-name">${f.feature}</span>
          <span class="bar-value">value = ${f.value.toFixed(3)}</span>
        </div>
        <div class="bar-track">
          <div class="bar-center-line"></div>
          <div class="bar-fill ${isPos ? 'pos' : 'neg'}" style="width:${pct}%"></div>
        </div>
        <div class="contribution-text">${f.shap_contribution > 0 ? '+' : ''}${f.shap_contribution.toFixed(4)} — ${f.direction}</div>
      </div>
    `;
  });

  panel.innerHTML = `
    <div class="verdict-row">
      <span class="verdict-label ${item.true_label}">${item.true_label === 'attack' ? '⚠ THREAT DETECTED' : '✓ BENIGN TRAFFIC'}</span>
    </div>
    <div class="meta">Test row #${item.test_row_index} · Model base rate: ${item.base_value.toFixed(4)} · Dataset: ${currentDataset.toUpperCase()}</div>
    <div class="section-label">Top 5 features driving this decision (SHAP)</div>
    ${barsHtml}
    <div class="explain-footer">
      Red bars push the model toward flagging this as an attack; green bars push toward benign.
      Bar length reflects the magnitude of that feature's contribution to this specific prediction —
      this is a local explanation for this one incident, not a global feature-importance ranking.
    </div>
  `;
}

document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentDataset = btn.dataset.dataset;
    currentIndex = 0;
    renderList();
    renderDetail();
  };
});

renderList();
renderDetail();
</script>

</body>
</html>
"""


def main():
    with open(UNSW_JSON) as f:
        unsw_data = json.load(f)
    with open(CICIDS_JSON) as f:
        cicids_data = json.load(f)

    combined = {"unsw": unsw_data, "cicids": cicids_data}
    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", json.dumps(combined))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(html)
    print(f"Dashboard built: {OUT_PATH}")


if __name__ == "__main__":
    main()
