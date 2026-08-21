# Canonical Files — What's Actually Live vs. Experimental History

This project has some file sprawl from iterative development (multiple
versions of federated learning, two RL implementations, several multiclass
classifier attempts). This document marks which file is the real,
currently-used one for each concept, so a reader doesn't have to guess.

## Legend
- ✅ CANONICAL — this is the real, currently-used implementation
- 🕰 SUPERSEDED — earlier iteration, kept for history/comparison, not used live
- ❓ NEEDS VERIFICATION — exists, purpose unclear, wasn't re-confirmed working this session
- ❌ CUT — implemented, tested, found invalid; excluded from final claims

## Data Preprocessing
- ✅ `data-collection/clean_data.py` — the real pipeline for both datasets (split → scale → SMOTE-on-train-only, verified leak-free)
- 🕰 `data-collection/clean_data_multiclass.py`, `clean_data_multiclass_cicids.py` — multiclass variants, not wired into the live API
- ❓ `data-collection/attack_taxonomy_mapping.py` — purpose not re-verified this session

## Detection Models (Layer 1 — Unsupervised)
- ✅ `ml-engine/isolation_forest.py` (UNSW) / `isolation_forest_cicids.py` (CIC-IDS)
- ✅ `ml-engine/lstm_autoencoder.py` (UNSW) / `autoencoder_cicids.py` (CIC-IDS)
  — **naming note:** despite the filename, this is a plain feedforward
  autoencoder, not an LSTM. Call it "Deep Autoencoder" in the report/demo.

## Detection Models (Layer 2 — Supervised)
- ✅ `ml-engine/random_forest.py` (UNSW) / `random_forest_cicids.py` (CIC-IDS)
- ✅ `ml-engine/xgboost_model.py` (UNSW) / `xgboost_cicids.py` (CIC-IDS)
- ✅ `ml-engine/tune_xgboost.py` — hyperparameter search (confirmed baseline was already near-optimal)
- ❓ `ml-engine/multiclass_classifier_unsw.py`, `multiclass_classifier_unsw_v2.py`, `multiclass_classifier_cicids.py` — 9-category classifiers exist, which version is canonical wasn't re-confirmed this session

## Ensemble
- ✅ `ml-engine/ensemble.py` (UNSW) — 91% acc/0.90 F1, confirmed
- ❓ `ml-engine/ensemble_cicids.py` vs `combine_ensemble_cicids.py` / `combine_ensemble_unsw.py` — duplicate-looking files, need one confirmed run to pick the canonical one

## Reduced Live-Feature Detector
- ✅ `ml-engine/train_live_detector.py` — the 7-feature model actually used by `/ingest_live` (UNSW F1=0.884, CIC-IDS F1=0.988)

## Online Learning / Self-Learning
- ✅ `ml-engine/online_learning_unsw.py` — F1 0.883→0.889
- ✅ `ml-engine/online_learning_cicids.py` — demonstrates catastrophic forgetting (F1 0.997→0.985)
- ✅ `ml-engine/online_learning_replay_cicids.py` — Experience Replay fix (forgetting cut ~75%)
- All three are canonical — together they tell one story (problem → fix), keep all three

## RL Response Engine
- ✅ `response-engine/train_policy.py` — **this is the one actually live in `api/main.py`**. Trained on real XGBoost confidence scores against real UNSW-NB15 labels. 23-state converged policy.
- 🕰 `response-engine/rl_response_agent.py` — separate, more elaborate synthetic-scenario implementation (8 threat types × severity × asset). Not wired into the API. Worth mentioning in the report as "an alternative RL formulation explored," but `train_policy.py` is the one actually running.
- 🕰 `response-engine/rl_response.py`, `integrated_pipeline_demo.py` — earlier iterations

## Shadow Copy Guardian
- ✅ `response-engine/shadow_copy_guardian.py` — real `tmutil` APFS snapshots for audit trail + file-level backup/restore for actual rollback (mounting live-root snapshots is blocked by macOS SIP — confirmed via testing, documented in the script's own docstring)
- 🕰 `response-engine/test_shadow_copy_guardian.py` — written for an earlier version of the script, doesn't match current logic; superseded by the manual bash burst-loop test used in final verification. Safe to delete or ignore.
- 🕰 `response-engine/simulate_ransomware_attack.py` — earlier test script

## Honeypot Grid
- ✅ `honeypot/monitor.py` — modify/delete/rename detection (reads intentionally excluded — macOS doesn't reliably report them)

## Federated Learning
- ✅ `federated/federated_learning_demo_v6_validated.py` — the final, client-validated version (automatically rejects degenerate nodes with insufficient positive-class data)
- 🕰 `federated_learning_demo.py`, `v2.py`, `v3.py`, `v4.py`, `v5_stress.py`, `federated_sim.py` — earlier iterations, kept for the comparison table already embedded in v6's own output (shows F1 across all versions side by side — genuinely useful for the report as an ablation story)

## Ransomware Lineage Tracker
- ✅ `lineage/ransomware_lineage.py` — TLSH clustering on **synthetic** mutated malware families (real malware samples deliberately avoided — correct call for a student project)

## Attack Prediction Engine
- ❌ `prediction/attack_prediction.py` — implemented and tested, but diagnosed as invalid: SMOTE/shuffle preprocessing destroys chronological ordering on both datasets, so "temporal windows" aren't real time. Excluded from final report claims; documented as future work.

## XAI Dashboards
- ✅ `xai-dashboard/app.py` — static SHAP dashboard (global importance + per-row explainer)
- ✅ `xai-dashboard/live_dashboard.py` — live, auto-refreshing SOC view, confirmed working (177 events, 89.8% accuracy screenshot-verified)
- ❓ `ml-engine/build_shap_dashboard.py`, `shap_explainer_cicids.py`, `shap_explainer_unsw.py` — likely helper/exploration scripts for the above, not independently re-verified

## Backend / Live System
- ✅ `api/main.py` — the real FastAPI backend: `/ingest`, `/ingest_live`, `/honeypot_alert`, `/incidents`, `/stats`
- ✅ `api/simulator.py` — test-set replay simulator (91.78% live accuracy, validated against offline metrics) — still useful for demo pacing even though `live_agent.py` supersedes it for genuine "live" claims
- ✅ `api/live_agent.py` — **the real one**: scapy packet capture → flow features → live detector → RL policy → MongoDB. Confirmed: 189 real captured packets processed correctly.
- ✅ `demo_full_pipeline.py` — single-entry-point terminal walkthrough of the full pipeline on one real test incident, no FastAPI/DB required

## Prediction helpers (role unclear, low priority)
- ❓ `ml-engine/predict_ae_cicids.py`, `predict_ae_unsw.py`, `predict_sklearn_cicids.py`, `predict_sklearn_unsw.py` — likely inference-testing scripts from development; the actual inference logic that matters is inlined directly in `api/main.py`

---
**For report writers / evaluators:** when in doubt about which file is "the real one," trust `api/main.py` and `api/live_agent.py` as the source of truth — they're what's actually running, and every ✅-marked file above is either imported by them directly or was independently run-and-verified in this project's build log.
