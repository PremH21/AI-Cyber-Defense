"""
Trains lightweight classifiers on a REDUCED, real-time-computable feature set
(duration, packet/byte counts, rate, port, protocol) for both datasets,
so the live packet-capture agent can query a real trained model using only
features it can actually derive from raw captured packets in real time.

This deliberately does NOT use the full 34/78-feature offline pipelines —
those need full-flow retrospective statistics (retransmission timing,
service-protocol lookups, etc.) that aren't practical to compute live.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
import joblib
import os

OUT_DIR = "ml-engine/models"
os.makedirs(OUT_DIR, exist_ok=True)


def train_unsw():
    print("=== UNSW-NB15: building reduced live-feature set ===")
    train_df = pd.read_parquet("data/processed/unsw-nb15_train.parquet")
    test_df = pd.read_parquet("data/processed/unsw-nb15_test.parquet")

    def build(df):
        out = pd.DataFrame()
        out["duration"] = df["dur"]
        out["total_packets"] = df["spkts"] + df["dpkts"]
        out["total_bytes"] = df["sbytes"] + df["dbytes"]
        out["packets_per_sec"] = df["rate"]
        out["avg_packet_size"] = (df["sbytes"] + df["dbytes"]) / out["total_packets"].replace(0, 1)
        out["dest_port"] = 0  # UNSW-NB15 doesn't expose raw port numbers post-preprocessing; placeholder
        out["protocol_tcp"] = (df["proto"] == df["proto"].mode()[0]).astype(int)  # best-effort proxy
        out["Attack"] = df["Attack"]
        return out

    tr, te = build(train_df), build(test_df)
    X_train, y_train = tr.drop(columns=["Attack"]), tr["Attack"]
    X_test, y_test = te.drop(columns=["Attack"]), te["Attack"]

    model = xgb.XGBClassifier(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1, eval_metric="logloss")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds, target_names=["benign", "attack"]))
    print(f"F1: {f1_score(y_test, preds):.4f}")

    joblib.dump(model, f"{OUT_DIR}/live_detector_unsw.joblib")
    print(f"Saved to {OUT_DIR}/live_detector_unsw.joblib\n")


def train_cicids():
    print("=== CIC-IDS-2017: building reduced live-feature set ===")
    train_df = pd.read_parquet("data/processed/cic-ids-2017_train.parquet")
    test_df = pd.read_parquet("data/processed/cic-ids-2017_test.parquet")

    def build(df):
        out = pd.DataFrame()
        out["duration"] = df["Flow Duration"] / 1e6  # microseconds -> seconds
        out["total_packets"] = df["Total Fwd Packets"] + df["Total Backward Packets"]
        out["total_bytes"] = df["Total Length of Fwd Packets"] + df["Total Length of Bwd Packets"]
        out["packets_per_sec"] = df["Flow Packets/s"].replace([np.inf, -np.inf], 0)
        out["avg_packet_size"] = df["Average Packet Size"]
        out["dest_port"] = df["Destination Port"]
        out["protocol_tcp"] = (df["SYN Flag Count"] > 0).astype(int)  # proxy for TCP vs UDP
        out["Attack"] = df["Attack"]
        return out

    tr, te = build(train_df), build(test_df)
    X_train, y_train = tr.drop(columns=["Attack"]), tr["Attack"]
    X_test, y_test = te.drop(columns=["Attack"]), te["Attack"]

    model = xgb.XGBClassifier(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1, eval_metric="logloss")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds, target_names=["benign", "attack"]))
    print(f"F1: {f1_score(y_test, preds):.4f}")

    joblib.dump(model, f"{OUT_DIR}/live_detector_cicids.joblib")
    print(f"Saved to {OUT_DIR}/live_detector_cicids.joblib\n")


if __name__ == "__main__":
    train_unsw()
    train_cicids()
