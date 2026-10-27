import json
from collections import Counter
import joblib
import numpy as np
from pymongo import MongoClient

M = "ml-engine/models"
feats = json.load(open(f"{M}/live_unsup_meta.json"))["features"]
scaler = joblib.load(f"{M}/live_unsup_scaler.joblib")
ae = joblib.load(f"{M}/live_autoencoder.joblib")
db = MongoClient("mongodb://localhost:27017")["cyberdefense"]

q = {"source": "live_capture", "timestamp": {"$type": ["double", "int", "long"]}}
docs = [d for d in db["flows"].find(q, {"_id": 0})
        if d.get("features") and all(k in d["features"] for k in feats)]
print("flows with all 16 features stored:", len(docs))
if not docs:
    raise SystemExit("stored 'features' do not hold the 16 model features; paste one flow document")
X = np.array([[d["features"][k] for k in feats] for d in docs], dtype=float)
Z = scaler.transform(np.log1p(np.clip(X, 0, None)))
E = (ae.predict(Z) - Z) ** 2


def flag(d, k):
    return bool(((d.get("ml") or {}).get("unsupervised") or {}).get(k))


both = np.array([flag(d, "iforest_flag") and flag(d, "autoencoder_flag") for d in docs])
print("flagged by both unsupervised models:", int(both.sum()))

print("\nscaled space the models see (train_std = std of log1p(feature) on CIC benign)")
print("%-18s %10s %9s %11s %10s %7s %12s" % ("feature", "train_mean", "train_std", "live_mean_z", "live_std_z", "|z|>4", "AE err share"))
for i, k in enumerate(feats):
    z = Z[:, i]
    print("%-18s %10.3f %9.3f %11.2f %10.2f %6.1f%% %11.1f%%" % (
        k, scaler.mean_[i], scaler.scale_[i], z.mean(), z.std(), 100 * (np.abs(z) > 4).mean(), 100 * E[:, i].sum() / E.sum()))

print("\nlargest |z| per feature on live flows:")
mx = np.abs(Z).max(axis=0)
for i in np.argsort(-mx)[:8]:
    j = int(np.argmax(np.abs(Z[:, i])))
    print("  %-18s max|z|=%7.1f  raw value=%s" % (feats[i], mx[i], X[j, i]))

print("\nfeature with the largest |z| inside each both-flagged flow:")
print(" ", dict(Counter(feats[int(np.argmax(np.abs(Z[i])))] for i in np.where(both)[0]).most_common()))

print("\nbehavioural detections during this capture:")
for d in db["flows"].find({"source": "live_capture", "verdict": "behavioural_detection"}, {"_id": 0}):
    ev = (d.get("explanation") or {}).get("evidence") or {}
    print(" ", d["attack_type"], d["dst_ip"], d["dst_port"], json.dumps(ev, default=str)[:300])
