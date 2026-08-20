import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

BASE_DIR = os.path.expanduser("~/ai-cyber-defense")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
ML_MODEL_DIR = os.path.join(BASE_DIR, "ml-engine", "models")
RESPONSE_MODEL_DIR = os.path.join(BASE_DIR, "response-engine", "models")

sys.path.insert(0, os.path.join(BASE_DIR, "response-engine"))
import rl_response_agent as rl  # reuses map_label_to_threat_type, severity_for_threat, ACTIONS

random.seed(7)
np.random.seed(7)


def load_q_table():
    with open(os.path.join(RESPONSE_MODEL_DIR, "q_table.json")) as f:
        raw = json.load(f)
    q_table = {}
    for key, q_values in raw.items():
        threat_type, severity, asset_level = key.split("|")
        q_table[(threat_type, int(severity), asset_level)] = q_values
    return q_table


def choose_action(q_table, threat_type, severity, asset_level):
    state = (threat_type, severity, asset_level)
    q_values = q_table.get(state)
    if q_values is None:
        # state never visited in training — fall back to nearest severity match
        candidates = [k for k in q_table if k[0] == threat_type and k[2] == asset_level]
        if candidates:
            nearest = min(candidates, key=lambda k: abs(k[1] - severity))
            q_values = q_table[nearest]
        else:
            q_values = {a: 0.0 for a in rl.ACTIONS}
    return max(q_values, key=q_values.get)


def main():
    print("=== Loading trained XGBoost multi-class classifier (CIC-IDS-2017) ===")
    xgb_model = joblib.load(os.path.join(ML_MODEL_DIR, "xgb_multiclass_cicids.joblib"))

    with open(os.path.join(DATA_DIR, "cicids_multiclass_labels.json")) as f:
        class_names = json.load(f)

    print("Loading test set...")
    test_df = pd.read_parquet(os.path.join(DATA_DIR, "cicids_multiclass_test.parquet"))
    y_true = test_df["AttackCat"].values
    X_test = test_df.drop(columns=["AttackCat"]).values

    print("Loading trained Q-table (RL response policy)...")
    q_table = load_q_table()
    print(f"Q-table has {len(q_table)} learned states\n")

    # Sample a diverse mix: some benign, some of each attack class present in test set
    sample_indices = []
    rng = np.random.default_rng(7)
    for class_idx in range(len(class_names)):
        idxs = np.where(y_true == class_idx)[0]
        if len(idxs) == 0:
            continue
        take = min(3, len(idxs))
        sample_indices.extend(rng.choice(idxs, size=take, replace=False))
    rng.shuffle(sample_indices)

    asset_levels_cycle = ["low", "medium", "high"]

    print(f"=== Running full DETECT -> CLASSIFY -> DECIDE -> RESPOND pipeline on {len(sample_indices)} real test incidents ===\n")

    latencies_ms = []
    action_counts = {}
    correct_classifications = 0

    for i, idx in enumerate(sample_indices):
        row = X_test[idx:idx+1]
        true_label = class_names[y_true[idx]]
        asset_level = asset_levels_cycle[i % 3]

        t0 = time.perf_counter()

        # DETECT + CLASSIFY
        dmatrix = xgb.DMatrix(row)
        pred_class_idx = int(xgb_model.predict(dmatrix)[0])
        pred_label = class_names[pred_class_idx]

        # DECIDE: map prediction -> threat_type -> severity -> RL action
        threat_type = rl.map_label_to_threat_type(pred_label)
        severity = rl.severity_for_threat(threat_type)
        action = choose_action(q_table, threat_type, severity, asset_level)

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000
        latencies_ms.append(latency_ms)

        action_counts[action] = action_counts.get(action, 0) + 1
        is_correct = (pred_label == true_label)
        correct_classifications += int(is_correct)

        match_flag = "OK" if is_correct else "MISCLASSIFIED"
        print(f"[{i+1:2d}] true={true_label:28s} pred={pred_label:28s} [{match_flag:13s}] "
              f"-> threat_type={threat_type:18s} sev={severity:2d} asset={asset_level:6s} "
              f"-> ACTION: {action:14s} ({latency_ms:.2f} ms)")

    print(f"\n=== Summary over {len(sample_indices)} real incidents ===")
    print(f"Classification accuracy on this sample: {correct_classifications}/{len(sample_indices)} "
          f"({100*correct_classifications/len(sample_indices):.1f}%)")
    print(f"\nAction distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action:14s}: {count}")

    print(f"\n=== HONEST LATENCY MEASUREMENT (detect+classify+decide, per incident) ===")
    print(f"Mean:   {np.mean(latencies_ms):.3f} ms")
    print(f"Median: {np.median(latencies_ms):.3f} ms")
    print(f"P95:    {np.percentile(latencies_ms, 95):.3f} ms")
    print(f"Max:    {np.max(latencies_ms):.3f} ms")
    print("\nNote: this measures classification + decision only (not network I/O, firewall")
    print("rule injection, or process termination, which would add real-world overhead).")
    print("Report this measured number, not an unverified target.")


if __name__ == "__main__":
    main()
