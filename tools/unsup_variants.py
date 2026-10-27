"""
Compare unsupervised-layer variants. Read-only on MongoDB and on your saved models.
New models go to ml-engine/models/variants/. Report goes to ml-engine/models/unsup_variants_report.json
Run from the project root:  python unsup_variants.py
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

M = "ml-engine/models"
FPR = 0.01
CAP = 300_000
EVAL_CAP = 100_000
FLAGS = ["syn_count", "fin_count", "rst_count", "psh_count", "ack_count"]


def prep(A):
    return np.log1p(np.clip(np.asarray(A, dtype=np.float64), 0, None))


def recon_err(ae, Z):
    return ((ae.predict(Z) - Z) ** 2).mean(axis=1)


class Variant:
    def __init__(self, name, cols, scaler, iso, ae, iso_t, ae_t):
        self.name, self.cols, self.scaler, self.iso, self.ae = name, cols, scaler, iso, ae
        self.iso_t, self.ae_t = iso_t, ae_t

    def rates(self, df):
        if len(df) == 0:
            return None
        Z = self.scaler.transform(prep(df[self.cols].values))
        i = self.iso.score_samples(Z) < self.iso_t
        a = recon_err(self.ae, Z) > self.ae_t
        return {"n": int(len(df)), "if": float(i.mean()), "ae": float(a.mean()),
                "either": float((i | a).mean()), "both": float((i & a).mean())}


def fit_variant(name, cols, ben_tr, ben_va):
    scaler = StandardScaler().fit(prep(ben_tr[cols].values))
    Z = scaler.transform(prep(ben_tr[cols].values))
    iso = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1).fit(Z)
    ae = MLPRegressor(hidden_layer_sizes=(12, 8, 4, 8, 12), activation="relu", solver="adam", batch_size=1024,
                      learning_rate_init=1e-3, max_iter=25, tol=1e-5, n_iter_no_change=5, random_state=42)
    ae.fit(Z, Z)
    Zv = scaler.transform(prep(ben_va[cols].values))
    iso_t = float(np.percentile(iso.score_samples(Zv), FPR * 100))
    ae_t = float(np.percentile(recon_err(ae, Zv), 100 - FPR * 100))
    return Variant(name, cols, scaler, iso, ae, iso_t, ae_t)


def print_tables(results, order, variants):
    names = [v for v in variants if v in results]
    for rule, label in [("both", "BOTH models flag (current engine rule)"), ("either", "EITHER model flags"),
                        ("ae", "autoencoder only")]:
        print(f"\n=== flag rate, rule: {label} ===")
        print("%-26s %7s " % ("class", "n") + " ".join("%-22s" % n[:22] for n in names))
        for cls in order:
            row = [results[n].get(cls) for n in names]
            if all(r is None for r in row):
                continue
            n_ = next(r["n"] for r in row if r)
            print("%-26s %7d " % (cls[:26], n_) + " ".join(
                "%-22s" % (("%.2f%%" % (100 * r[rule])) if r else "-") for r in row))


def main():
    sys.path.insert(0, "ml-engine")
    from train_live_multiclass import load
    from pymongo import MongoClient

    meta = json.load(open(f"{M}/live_unsup_meta.json"))
    feats = meta["features"]
    df = load()
    y = df["Label"].to_numpy(dtype=object)
    X = df.drop(columns=["Label"])[feats]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
    ben_tr = X_tr[y_tr == "BENIGN"].sample(min(CAP, int((y_tr == "BENIGN").sum())), random_state=1)
    ben_va_all = X_va[y_va == "BENIGN"]
    ben_va = ben_va_all.sample(min(EVAL_CAP, len(ben_va_all)), random_state=2)
    ben_te_all = X_te[y_te == "BENIGN"]
    ben_te = ben_te_all.sample(min(EVAL_CAP, len(ben_te_all)), random_state=3)

    db = MongoClient("mongodb://localhost:27017")["cyberdefense"]
    q = {"source": "live_capture", "timestamp": {"$type": ["double", "int", "long"]}}
    rows = [d["features"] for d in db["flows"].find(q, {"_id": 0, "features": 1})
            if d.get("features") and all(k in d["features"] for k in feats)]
    live = pd.DataFrame(rows)[feats].astype(float)
    for c in FLAGS:
        live[c] = (live[c] > 0).astype(float)
    print("live flows used:", len(live), "| CIC benign train/val/test:", len(ben_tr), len(ben_va), len(ben_te))

    variants = [Variant("V0 saved models", feats, joblib.load(f"{M}/live_unsup_scaler.joblib"),
                        joblib.load(f"{M}/live_iforest.joblib"), joblib.load(f"{M}/live_autoencoder.joblib"),
                        meta["iso_threshold"], meta["ae_threshold"])]
    for name, drop in [("V2a drop rst", ["rst_count"]), ("V2b drop rst+fin", ["rst_count", "fin_count"])]:
        print("training", name, "...", flush=True)
        variants.append(fit_variant(name, [c for c in feats if c not in drop], ben_tr, ben_va))
    os.makedirs(f"{M}/variants", exist_ok=True)
    for v in variants[1:]:
        tag = v.name.split()[0]
        joblib.dump({"cols": v.cols, "scaler": v.scaler, "iso": v.iso, "ae": v.ae, "iso_t": v.iso_t, "ae_t": v.ae_t},
                    f"{M}/variants/{tag}.joblib")

    attack_classes = sorted(c for c in set(y_te) if c != "BENIGN")
    results = {}
    for v in variants:
        r = {"BENIGN (CIC test)": v.rates(ben_te), "LIVE capture": v.rates(live)}
        for c in attack_classes:
            r[c] = v.rates(X_te[y_te == c])
        results[v.name] = r
    order = ["BENIGN (CIC test)"] + attack_classes + ["LIVE capture"]
    print_tables(results, order, [v.name for v in variants])
    json.dump(results, open(f"{M}/unsup_variants_report.json", "w"), indent=2)
    print("\nThresholds were set for 1% flagged on CIC benign validation. Saved report and variants.")


if __name__ == "__main__":
    main()
