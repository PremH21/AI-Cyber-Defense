"""
Trains the LIVE detector from RAW (unscaled) CIC-IDS-2017 flows, using only
features the packet-capture agent can compute from real traffic.
One multiclass XGBoost model yields: attack category, and
attack probability = 1 - P(BENIGN). No SMOTE, no scaling (trees don't need it),
so training-time and live-time feature values are on the same scale.

Run from the project root:  python ml-engine/train_live_multiclass.py
"""
import glob
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

CSV_GLOB = "data/cic-ids-2017/*.csv"
OUT = "ml-engine/models"

# raw CIC-IDS column -> live feature name (all computable from live packets)
FEATURES = {
    "Flow Duration": "duration_us",
    "Total Fwd Packets": "fwd_packets",
    "Total Backward Packets": "bwd_packets",
    "Total Length of Fwd Packets": "fwd_bytes",
    "Total Length of Bwd Packets": "bwd_bytes",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "Average Packet Size": "avg_packet_size",
    "Flow Bytes/s": "bytes_per_sec",
    "Flow Packets/s": "packets_per_sec",
    "Destination Port": "dst_port",
    "SYN Flag Count": "syn_count",
    "FIN Flag Count": "fin_count",
    "RST Flag Count": "rst_count",
    "PSH Flag Count": "psh_count",
    "ACK Flag Count": "ack_count",
}


def load():
    paths = sorted(glob.glob(CSV_GLOB))
    if not paths:
        raise SystemExit(f"No CSV files found at {CSV_GLOB}. Run from ~/ai-cyber-defense.")
    frames = []
    for path in paths:
        print("Reading", os.path.basename(path), flush=True)
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df[list(FEATURES) + ["Label"]])
    df = pd.concat(frames, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
    df["Label"] = (
        df["Label"].astype(str).str.strip().apply(lambda s: re.sub(r"[^\x00-\x7F]+", "-", s))
    )
    df = df.rename(columns=FEATURES)
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load()
    classes = sorted(df["Label"].unique())
    cid = {c: i for i, c in enumerate(classes)}
    y = df["Label"].map(cid).values
    X = df.drop(columns=["Label"])
    print(f"\n{len(X)} flows, {X.shape[1]} features, {len(classes)} classes")
    print(df["Label"].value_counts().to_string(), "\n")

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

    # sqrt-balanced sample weights: helps rare classes without exploding on tiny ones
    counts = np.bincount(y_tr, minlength=len(classes)).astype(float)
    w_class = np.sqrt(counts.sum() / (len(classes) * np.maximum(counts, 1)))
    sw = w_class[y_tr]

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        n_estimators=250,
        max_depth=8,
        learning_rate=0.1,
        tree_method="hist",
        n_jobs=-1,
        eval_metric="mlogloss",
        early_stopping_rounds=15,
        random_state=42,
    )
    model.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_va, y_va)], verbose=25)

    proba = model.predict_proba(X_te)
    pred = proba.argmax(axis=1)
    benign_id = cid["BENIGN"]
    y_bin = (y_te != benign_id).astype(int)
    p_bin = (pred != benign_id).astype(int)

    report = classification_report(
        y_te, pred, labels=list(range(len(classes))), target_names=classes, zero_division=0
    )
    summary = (
        f"Held-out TEST: {len(y_te)} flows\n"
        f"Multiclass macro-F1: {f1_score(y_te, pred, average='macro'):.4f}\n"
        f"Binary (attack vs benign) F1: {f1_score(y_bin, p_bin):.4f}\n\n{report}"
    )
    print("\n" + summary)

    joblib.dump(model, f"{OUT}/live_multiclass_cicids.joblib")
    X_tr.sample(min(500, len(X_tr)), random_state=1).to_parquet(f"{OUT}/live_shap_background.parquet")
    with open(f"{OUT}/live_multiclass_meta.json", "w") as f:
        json.dump({"classes": classes, "features": list(X.columns)}, f, indent=2)
    with open(f"{OUT}/live_multiclass_report.txt", "w") as f:
        f.write(summary)
    print(f"Saved model, meta, SHAP background, report to {OUT}/")


if __name__ == "__main__":
    main()
