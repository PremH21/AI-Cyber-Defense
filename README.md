# AI-Driven Autonomous Defense Framework

A self-learning cybersecurity platform combining unsupervised anomaly
detection, supervised threat classification, reinforcement-learning-based
automated response, federated learning, and explainable AI — built and
validated end-to-end, including genuine real-time network packet capture.

**BLDEA's Dr. P G Halakatti College of Engineering & Technology, Bijapur**
Dept. of CSE (AI & ML) — Major Project 2025–26

## What actually works (verified, not aspirational)

| Capability | Status | Evidence |
|---|---|---|
| Zero-day detection (Isolation Forest + Autoencoder) | Done | UNSW: 60%/69% acc, CIC-IDS: trained, validated |
| Known-threat classification (RF + XGBoost) | Done | UNSW: F1=0.90, CIC-IDS: F1≈0.99 (verified not data leakage) |
| Weighted ensemble | Done | 91% acc / 0.90 F1 on held-out test |
| Reduced live-feature detector | Done | UNSW F1=0.884, CIC-IDS F1=0.988 - powers real-time capture |
| SHAP explainability | Done | Static + live per-incident dashboard |
| RL response policy | Done | Q-learning, 23-state converged tiered policy (alert to block/quarantine/rollback by severity) |
| Federated learning (FedAvg) | Done | 4 nodes, automatic rejection of degenerate/poisoned clients |
| Online learning + Experience Replay | Done | Proved catastrophic forgetting (F1 0.997 to 0.985) and that replay cuts it ~75% |
| Honeypot Grid | Done | Real file-watcher, modify/delete/rename detection |
| Shadow Copy Guardian | Done | Real APFS snapshots (audit trail) + confirmed working file-level rollback |
| Ransomware Lineage Tracker | Done | TLSH clustering on synthetic malware family simulation |
| Live packet capture | Done | Real scapy sniffing on a live network interface, real model, real RL decision, MongoDB. 189 real captured events confirmed. |
| Attack Prediction Engine | Cut | Implemented, tested, found invalid (SMOTE/shuffle preprocessing destroys timestamp ordering on both datasets) - documented as future work |

See `CANONICAL_FILES.md` for which specific file is the real implementation
of each item above (several were built iteratively and old versions were kept).

## Architecture
Real network traffic (scapy capture)
|
v
Flow feature extraction (7 real-time-computable features)
|
v
Detection Engine
Layer 1 (unsupervised): Isolation Forest + Autoencoder -> zero-day signal
Layer 2 (supervised): Random Forest + XGBoost -> known-threat classification
Ensemble vote (weighted: RF 0.35, XGB 0.35, IF 0.15, AE 0.15)
|
v
SHAP explanation (top-5 contributing features per decision)
|
v
Decision Engine: severity score (1-10) + asset criticality
|
v
RL Response Policy (trained Q-table) -> alert_only / block_ip / quarantine / rollback
|
v
MongoDB persistence <--- also written by: Honeypot Grid, Shadow Copy Guardian
|
v
Live SOC Dashboard (Streamlit, auto-refreshing)
Separately: Federated Learning (4-node FedAvg simulation) and
Online Learning + Experience Replay validate the self-learning claims
independently of the live pipeline above.

## Running the full live system

Requires: macOS, conda environment `cyberdefense` (Python 3.11, arm64),
Docker Desktop running.

**Terminal 1 - databases:**
```bash
cd ~/ai-cyber-defense
conda activate cyberdefense
docker-compose up -d
```

**Terminal 2 - backend API:**
```bash
cd ~/ai-cyber-defense
conda activate cyberdefense
uvicorn api.main:app --reload --port 8000
```
Wait for `All models loaded` and `Connected to MongoDB`.

**Terminal 3 - live packet capture (requires sudo for raw socket access):**
```bash
cd ~/ai-cyber-defense
conda activate cyberdefense
sudo $(which python) api/live_agent.py
```

**Terminal 4 - live dashboard:**
```bash
cd ~/ai-cyber-defense
conda activate cyberdefense
streamlit run xai-dashboard/live_dashboard.py --server.port 8501
```
Open `http://localhost:8501`.

**Optional Terminal 5 - Honeypot Grid:**
```bash
cd ~/ai-cyber-defense
conda activate cyberdefense
python honeypot/monitor.py
```

**Optional Terminal 6 - Shadow Copy Guardian (requires sudo):**
```bash
cd ~/ai-cyber-defense
conda activate cyberdefense
sudo $(which python) response-engine/shadow_copy_guardian.py
```

## Quick single-file demo (no servers required)
```bash
python demo_full_pipeline.py
```
Walks through one real test incident end-to-end in the terminal: detection,
classification, SHAP explanation, RL decision.

## Datasets
- UNSW-NB15 (2.5M records, 9 attack categories)
- CIC-IDS-2017 (2.8M flows - DDoS, PortScan, brute-force, web attacks, infiltration)

## Known limitations (stated honestly)
- Live capture runs locally on the machine/network being protected - this is
  true of any real EDR/XDR tool, not a limitation specific to this project.
- Shadow Copy Guardian's rollback uses file-level backup rather than mounting
  the OS-level APFS snapshot directly, because macOS's System Integrity
  Protection blocks mounting the live root volume - confirmed via testing.
  The OS snapshot is still taken for a genuine audit trail.
- Attack Prediction Engine was cut after diagnosis (see table above).
- Some file sprawl exists from iterative development - see `CANONICAL_FILES.md`.
