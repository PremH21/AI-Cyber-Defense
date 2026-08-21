"""
AI-Driven Autonomous Defense Framework — Backend API

Wraps the real trained models (Isolation Forest, Autoencoder, Random Forest,
XGBoost, RL policy table) behind one live ingestion endpoint. Every
processed incident is persisted to MongoDB.

Run with: uvicorn api.main:app --reload --port 8000
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
import time
from datetime import datetime
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient, DESCENDING

MODELS = {}
MONGO_CLIENT = None
DB = None

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "cyberdefense"
INCIDENTS_COLLECTION = "incidents"


def load_models():
    print("Loading trained models into memory...")
    MODELS["iso_forest"] = joblib.load("ml-engine/models/isolation_forest_unsw.joblib")
    MODELS["rf"] = joblib.load("ml-engine/models/random_forest_unsw.joblib")
    MODELS["xgb"] = joblib.load("ml-engine/models/xgboost_unsw.joblib")

    import torch, torch.nn as nn

    class Autoencoder(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 8), nn.ReLU(),
                nn.Linear(8, 16), nn.ReLU(),
                nn.Linear(16, 32), nn.ReLU(),
                nn.Linear(32, n_features)
            )
        def forward(self, x): return self.net(x)

    ae_meta = joblib.load("ml-engine/models/lstm_ae_meta_unsw.joblib")
    ae = Autoencoder(ae_meta["n_features"])
    ae.load_state_dict(torch.load("ml-engine/models/lstm_autoencoder_unsw.pt", map_location="cpu"))
    ae.eval()
    MODELS["autoencoder"] = ae
    MODELS["ae_threshold"] = 0.04192
    MODELS["torch"] = torch

    MODELS["shap_explainer"] = shap.TreeExplainer(MODELS["xgb"])

    with open("response-engine/models/q_table.json") as f:
        MODELS["q_table"] = json.load(f)

    MODELS["test_df"] = pd.read_parquet("data/processed/unsw-nb15_test.parquet")
    print(f"All models loaded. Test set: {len(MODELS['test_df'])} rows available for ingestion.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MONGO_CLIENT, DB
    load_models()
    MONGO_CLIENT = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    DB = MONGO_CLIENT[DB_NAME]
    try:
        MONGO_CLIENT.server_info()
        print(f"Connected to MongoDB at {MONGO_URI}")
    except Exception as e:
        print(f"WARNING: MongoDB not reachable ({e}). Incidents will not be persisted.")
    yield
    if MONGO_CLIENT:
        MONGO_CLIENT.close()


app = FastAPI(title="AI-Driven Autonomous Defense Framework API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class IngestRequest(BaseModel):
    row_index: int
    asset_criticality: str = "medium"  # low | medium | high — real deployments would derive this from asset inventory


def get_action(severity: int, asset: str) -> str:
    state = f"{severity}|{asset}"
    q_values = MODELS["q_table"].get(state)
    if q_values is None:
        # fallback: try same severity with any asset tier before defaulting
        for a in ("medium", "low", "high"):
            alt = MODELS["q_table"].get(f"{severity}|{a}")
            if alt:
                q_values = alt
                break
    if q_values is None:
        return "alert_only"
    return max(q_values, key=q_values.get)


def run_pipeline(row_index: int, asset_criticality: str):
    test_df = MODELS["test_df"]
    if row_index < 0 or row_index >= len(test_df):
        raise HTTPException(400, f"row_index out of range (0-{len(test_df)-1})")
    if asset_criticality not in ("low", "medium", "high"):
        raise HTTPException(400, "asset_criticality must be low, medium, or high")

    true_label = int(test_df.iloc[row_index]["Attack"])
    incident = test_df.drop(columns=["Attack"]).iloc[[row_index]]

    t0 = time.time()

    # Layer 2: supervised classification (primary detection)
    xgb_pred = int(MODELS["xgb"].predict(incident)[0])
    xgb_proba = float(MODELS["xgb"].predict_proba(incident)[0][1])
    rf_pred = int(MODELS["rf"].predict(incident)[0])

    # Layer 1: unsupervised zero-day flags
    iso_flag = bool(MODELS["iso_forest"].predict(incident)[0] == -1)
    torch = MODELS["torch"]
    X_t = torch.tensor(incident.values.astype(np.float32))
    with torch.no_grad():
        ae_err = float(torch.mean((MODELS["autoencoder"](X_t) - X_t) ** 2))
    ae_flag = ae_err > MODELS["ae_threshold"]

    # Ensemble vote (same weights as validated in ensemble.py)
    weighted = 0.35 * rf_pred + 0.35 * xgb_pred + 0.15 * int(iso_flag) + 0.15 * int(ae_flag)
    final_pred = int(weighted >= 0.5)

    detect_ms = (time.time() - t0) * 1000

    # XAI explanation
    shap_values = MODELS["shap_explainer"].shap_values(incident)[0]
    feature_names = incident.columns.tolist()
    top5_idx = np.argsort(np.abs(shap_values))[::-1][:5]
    top5 = [
        {"feature": feature_names[i], "value": float(incident.iloc[0, i]),
         "shap_contribution": float(shap_values[i]),
         "direction": "attack" if shap_values[i] > 0 else "benign"}
        for i in top5_idx
    ]

    # Decision: severity + RL action
    severity = min(10, max(1, int(xgb_proba * 10) + (2 if final_pred else 0)))
    action = get_action(severity, asset_criticality)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "row_index": row_index,
        "true_label": "attack" if true_label == 1 else "benign",
        "predicted_label": "attack" if final_pred == 1 else "benign",
        "xgb_confidence": round(xgb_proba, 4),
        "zero_day_flags": {"isolation_forest": iso_flag, "autoencoder": ae_flag},
        "detection_latency_ms": round(detect_ms, 3),
        "top_5_features": top5,
        "severity": severity,
        "asset_criticality": asset_criticality,
        "action_taken": action,
        "correct": bool(final_pred == true_label),
    }

    if DB is not None:
        try:
            DB[INCIDENTS_COLLECTION].insert_one(dict(record))
        except Exception as e:
            print(f"Mongo insert failed: {e}")

    return record


@app.get("/")
def root():
    return {"service": "AI-Driven Autonomous Defense Framework", "status": "running",
            "models_loaded": [k for k in MODELS.keys() if k not in ("test_df", "torch")]}


@app.post("/ingest")
def ingest(req: IngestRequest):
    return run_pipeline(req.row_index, req.asset_criticality)


@app.get("/incidents")
def get_incidents(limit: int = 50):
    if DB is None:
        raise HTTPException(503, "Database not connected")
    docs = list(DB[INCIDENTS_COLLECTION].find({}, {"_id": 0}).sort("timestamp", DESCENDING).limit(limit))
    return {"count": len(docs), "incidents": docs}


@app.get("/stats")
def get_stats():
    if DB is None:
        raise HTTPException(503, "Database not connected")
    total = DB[INCIDENTS_COLLECTION].count_documents({})
    correct = DB[INCIDENTS_COLLECTION].count_documents({"correct": True})
    attacks = DB[INCIDENTS_COLLECTION].count_documents({"predicted_label": "attack"})
    avg_cursor = DB[INCIDENTS_COLLECTION].aggregate([{"$group": {"_id": None, "avg_ms": {"$avg": "$detection_latency_ms"}}}])
    avg_latency = next(avg_cursor, {}).get("avg_ms", 0)
    return {"total_incidents_processed": total,
            "accuracy": round(correct / total, 4) if total else None,
            "attacks_detected": attacks,
            "avg_detection_latency_ms": round(avg_latency, 3) if avg_latency else None}


class HoneypotAlert(BaseModel):
    decoy_file: str
    event_type: str  # "accessed" | "modified" | "deleted"


@app.post("/honeypot_alert")
def honeypot_alert(alert: HoneypotAlert):
    """
    Honeypot events are zero-false-positive by design — no legitimate process
    should ever touch a decoy file, so any event here is treated as a
    confirmed attack at maximum severity, routed straight to the RL policy
    for a real containment action.
    """
    severity = 10
    action = get_action(severity, "high")

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "source": "honeypot",
        "decoy_file": alert.decoy_file,
        "event_type": alert.event_type,
        "true_label": "attack",
        "predicted_label": "attack",
        "severity": severity,
        "asset_criticality": "high",
        "action_taken": action,
        "correct": True,
        "detection_latency_ms": 0.0,
    }

    if DB is not None:
        try:
            DB[INCIDENTS_COLLECTION].insert_one(dict(record))
        except Exception as e:
            print(f"Mongo insert failed: {e}")

    print(f"HONEYPOT TRIGGERED: {alert.decoy_file} was {alert.event_type} — action: {action}")
    return record


class LiveFlowRequest(BaseModel):
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    ip_version: str
    duration_sec: float
    packet_count: int
    total_bytes: int
    avg_packet_size: float
    packets_per_sec: float


@app.post("/ingest_live")
def ingest_live(flow: LiveFlowRequest):
    """
    Real-time endpoint for the live packet-capture agent. Uses the reduced
    7-feature live detector (trained on the same statistical signal, but
    only features genuinely computable from live traffic without full-flow
    retrospective analysis).
    """
    if "live_detector" not in MODELS:
        MODELS["live_detector"] = joblib.load("ml-engine/models/live_detector_cicids.joblib")

    features = pd.DataFrame([{
        "duration": flow.duration_sec,
        "total_packets": flow.packet_count,
        "total_bytes": flow.total_bytes,
        "packets_per_sec": flow.packets_per_sec,
        "avg_packet_size": flow.avg_packet_size,
        "dest_port": flow.dst_port,
        "protocol_tcp": 1 if flow.protocol == "TCP" else 0,
    }])

    t0 = time.time()
    proba = float(MODELS["live_detector"].predict_proba(features)[0][1])
    pred = int(proba >= 0.5)
    detect_ms = (time.time() - t0) * 1000

    severity = min(10, max(1, int(proba * 10)))
    action = get_action(severity, "medium")  # live traffic: default to medium criticality

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "source": "live_capture",
        "src_ip": flow.src_ip, "dst_ip": flow.dst_ip,
        "src_port": flow.src_port, "dst_port": flow.dst_port,
        "protocol": flow.protocol, "ip_version": flow.ip_version,
        "predicted_label": "attack" if pred == 1 else "benign",
        "confidence": round(proba, 4),
        "severity": severity,
        "action_taken": action,
        "detection_latency_ms": round(detect_ms, 3),
    }

    if DB is not None:
        try:
            DB[INCIDENTS_COLLECTION].insert_one(dict(record))
        except Exception as e:
            print(f"Mongo insert failed: {e}")

    return record
