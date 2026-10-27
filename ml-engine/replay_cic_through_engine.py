"""
End-to-end evaluation of the engine on REAL held-out CIC-IDS-2017 flows.

Re-creates the exact held-out test split used in training (same seed), samples flows
per attack class, and runs each through the full engine: supervised -> unsupervised ->
severity -> SHAP -> RL action. Reports, per class: how often the engine raised an alert,
how often it named the right attack type, average severity, most common action, latency.

Run from the project root:  python ml-engine/replay_cic_through_engine.py
"""
import json
import os
import sys
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, "ml-engine")
sys.path.insert(0, "api")
from train_live_multiclass import load  # noqa: E402
from engine import DefenseEngine, TAXONOMY  # noqa: E402

PER_CLASS = int(os.environ.get("PER_CLASS", "200"))


def main():
    eng = DefenseEngine()
    df = load()
    y = df["Label"].map({c: i for i, c in enumerate(eng.classes)}).values
    X = df.drop(columns=["Label"])
    _, X_tmp, _, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    _, X_te, _, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
    print(f"Held-out test flows: {len(X_te)}")

    rng = np.random.RandomState(7)
    lines, sample_alert = [], None
    tot_lat, tot_n = 0.0, 0
    for ci, cls in enumerate(eng.classes):
        idx = np.where(y_te == ci)[0]
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, min(PER_CLASS, len(idx)), replace=False)
        expected = TAXONOMY.get(cls, (cls,))[0]
        alerts = correct = 0
        sev, acts, lat, cats = [], Counter(), [], Counter()
        for j in pick:
            row = X_te.iloc[j]
            fv = {k: float(row[k]) for k in eng.features}
            flow = {"src_ip": f"10.1.{j % 250}.{(j // 250) % 250 + 1}", "dst_ip": "192.168.1.10",
                    "src_port": 40000, "dst_port": int(fv["dst_port"]), "protocol": "TCP",
                    "start_ts": 1000.0 + j, "duration_sec": fv["duration_us"] / 1e6,
                    "fwd_packets": int(fv["fwd_packets"]), "bwd_packets": int(fv["bwd_packets"]),
                    "fwd_bytes": fv["fwd_bytes"], "bwd_bytes": fv["bwd_bytes"]}
            r = eng.analyze_flow(flow, features_override=fv, use_tracker=False)
            lat.append(r["latency_ms"])
            if r["predicted_label"] == "attack":
                alerts += 1
                sev.append(r["severity"])
                acts[r["action_taken"]] += 1
                cats[r["attack_type"]] += 1
                if r["attack_type"] == expected:
                    correct += 1
                if sample_alert is None and cls != "BENIGN" and r["explanation"] and r["explanation"]["type"] == "shap":
                    sample_alert = r
        n = len(pick)
        tot_lat += sum(lat)
        tot_n += n
        if cls == "BENIGN":
            lines.append(f"{cls:18s} n={n:4d}  FALSE ALARM rate {alerts / n:6.1%}   (benign flows wrongly alerted)")
        else:
            lines.append(f"{cls:18s} n={n:4d}  alerted {alerts / n:6.1%}  correct type {correct / n:6.1%}  "
                         f"avg severity {np.mean(sev) if sev else 0:4.1f}  top action "
                         f"{acts.most_common(1)[0][0] if acts else '-':13s} latency {np.mean(lat):5.1f} ms  "
                         f"top label: {cats.most_common(1)[0][0] if cats else '-'}")
        print(lines[-1], flush=True)

    report = "\n".join(lines) + f"\n\nMean end-to-end latency per flow (incl. SHAP): {tot_lat / tot_n:.1f} ms over {tot_n} flows\n"
    open("ml-engine/models/engine_replay_report.txt", "w").write(report)
    print("\n" + report)
    if sample_alert:
        sample_alert.pop("features", None)
        print("Sample alert with SHAP explanation:\n" + json.dumps(sample_alert, indent=2, default=str)[:2200])


if __name__ == "__main__":
    main()
