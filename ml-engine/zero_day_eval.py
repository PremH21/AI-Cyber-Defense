"""
Zero-day evaluation by leave-one-family-out, plus production unsupervised layer.

- Unsupervised models (Isolation Forest, Deep Autoencoder) train on BENIGN flows
  only, so they never see any attack family -> trained once, saved for the live engine.
- For each held-out family, a supervised XGBoost is trained WITHOUT that family.
  We then measure how often each layer catches the unseen family, and the
  false-positive rate on benign test traffic.

Run from the project root:  python ml-engine/zero_day_eval.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from train_live_multiclass import load

OUT = "ml-engine/models"
HOLDOUTS = ["DoS Hulk", "PortScan", "DDoS", "FTP-Patator", "Bot", "Web Attack - Brute Force"]
BENIGN_CAP = 300_000
EVAL_CAP = 100_000
FPR_TARGET = 0.01  # thresholds set so ~1% of benign validation flows are flagged
np.random.seed(42)


def prep(X):
    """Heavy-tailed raw features -> log scale. The live engine must reuse exactly this."""
    return np.log1p(np.clip(X.values.astype(np.float64), 0, None))


def ae_error(ae, Z):
    return ((ae.predict(Z) - Z) ** 2).mean(axis=1)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load()
    y = df["Label"].to_numpy(dtype=object)
    X = df.drop(columns=["Label"])
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

    # ---------- unsupervised layer: benign only ----------
    n_ben = int((y_tr == "BENIGN").sum())
    ben_tr = X_tr[y_tr == "BENIGN"].sample(min(BENIGN_CAP, n_ben), random_state=1)
    scaler = StandardScaler().fit(prep(ben_tr))
    Z_tr = scaler.transform(prep(ben_tr))

    print("Training Isolation Forest on benign flows...", flush=True)
    iso = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1).fit(Z_tr)

    print("Training Deep Autoencoder (16-12-8-4-8-12-16) on benign flows...", flush=True)
    ae = MLPRegressor(hidden_layer_sizes=(12, 8, 4, 8, 12), activation="relu", solver="adam",
                      batch_size=1024, learning_rate_init=1e-3, max_iter=25, tol=1e-5,
                      n_iter_no_change=5, random_state=42, verbose=True)
    ae.fit(Z_tr, Z_tr)

    ben_va_all = X_va[y_va == "BENIGN"]
    ben_va = ben_va_all.sample(min(EVAL_CAP, len(ben_va_all)), random_state=2)
    Zv = scaler.transform(prep(ben_va))
    iso_thr = float(np.percentile(iso.score_samples(Zv), FPR_TARGET * 100))   # below = anomalous
    ae_thr = float(np.percentile(ae_error(ae, Zv), 100 - FPR_TARGET * 100))   # above = anomalous
    print(f"Thresholds at {FPR_TARGET:.0%} benign FPR: IF score < {iso_thr:.4f}, AE error > {ae_thr:.4f}")

    def unsup(Xdf):
        Z = scaler.transform(prep(Xdf))
        return iso.score_samples(Z) < iso_thr, ae_error(ae, Z) > ae_thr

    joblib.dump(iso, f"{OUT}/live_iforest.joblib")
    joblib.dump(ae, f"{OUT}/live_autoencoder.joblib")
    joblib.dump(scaler, f"{OUT}/live_unsup_scaler.joblib")
    with open(f"{OUT}/live_unsup_meta.json", "w") as f:
        json.dump({"iso_threshold": iso_thr, "ae_threshold": ae_thr, "features": list(X.columns),
                   "benign_fpr_target": FPR_TARGET, "transform": "log1p(clip(x,0)) then StandardScaler"},
                  f, indent=2)

    # ---------- benign test false-positive rates ----------
    ben_te_all = X_te[y_te == "BENIGN"]
    ben_te = ben_te_all.sample(min(EVAL_CAP, len(ben_te_all)), random_state=3)
    ib, ab = unsup(ben_te)
    print(f"\nBenign test FPR: IF {ib.mean():.2%} | AE {ab.mean():.2%} | "
          f"either {np.mean(ib | ab):.2%} | both {np.mean(ib & ab):.2%}\n")

    # ---------- leave-one-family-out ----------
    rows = []
    for fam in HOLDOUTS:
        unseen = X_te[y_te == fam]
        if len(unseen) == 0:
            print(f"Skipping '{fam}': not found in test split")
            continue
        print(f"=== Hiding '{fam}' from the supervised model ({len(unseen)} test flows) ===", flush=True)
        keep = y_tr != fam
        Xs, ys = X_tr[keep], y_tr[keep]
        ben_idx = np.where(ys == "BENIGN")[0]
        chosen = np.random.RandomState(0).choice(ben_idx, min(BENIGN_CAP, len(ben_idx)), replace=False)
        mask = (ys != "BENIGN")
        mask[chosen] = True
        Xs, ys = Xs[mask], ys[mask]
        classes = sorted(set(ys))
        cid = {c: i for i, c in enumerate(classes)}
        sup = xgb.XGBClassifier(objective="multi:softprob", n_estimators=100, max_depth=8,
                                tree_method="hist", n_jobs=-1, random_state=42)
        sup.fit(Xs, np.array([cid[c] for c in ys]))

        benign_id = cid["BENIGN"]
        sup_u = sup.predict(unseen) != benign_id
        sup_b = sup.predict(ben_te) != benign_id
        iu, au = unsup(unseen)
        rows.append({
            "held_out_family": fam, "n_flows": len(unseen),
            "supervised_only": sup_u.mean(),
            "iforest_only": iu.mean(), "autoencoder_only": au.mean(),
            "sup_OR_either_unsup": np.mean(sup_u | iu | au),
            "sup_OR_both_unsup": np.mean(sup_u | (iu & au)),
            "benign_FPR_either": np.mean(sup_b | ib | ab),
            "benign_FPR_both": np.mean(sup_b | (ib & ab)),
        })

    res = pd.DataFrame(rows).set_index("held_out_family")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print("\n=== Detection rate of an attack family the supervised model NEVER saw ===")
    print((res.drop(columns=["n_flows"]) * 100).round(1).astype(str) + "%")
    res.to_json(f"{OUT}/zero_day_eval.json", indent=2)
    print(f"\nSaved {OUT}/zero_day_eval.json and production unsupervised models.")


if __name__ == "__main__":
    main()
