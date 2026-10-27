import glob, json
from collections import Counter
import joblib
import numpy as np
import pandas as pd
from pymongo import MongoClient

M = "ml-engine/models"
meta = json.load(open(f"{M}/live_unsup_meta.json"))
feats, iso_t, ae_t = meta["features"], meta["iso_threshold"], meta["ae_threshold"]
scaler = joblib.load(f"{M}/live_unsup_scaler.joblib")
iso = joblib.load(f"{M}/live_iforest.joblib")
ae = joblib.load(f"{M}/live_autoencoder.joblib")
xg = joblib.load(f"{M}/live_multiclass_cicids.joblib")
benign = json.load(open(f"{M}/live_multiclass_meta.json"))["classes"].index("BENIGN")

FLAGS = ["FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count"]
KEY = {"FIN Flag Count": "fin_count", "SYN Flag Count": "syn_count", "RST Flag Count": "rst_count",
       "PSH Flag Count": "psh_count", "ACK Flag Count": "ack_count"}

print("=== 1. Raw CIC-IDS-2017 flag columns ===")
raw = {}
for f in sorted(glob.glob("data/cic-ids-2017/*.csv")):
    tag = "Monday" if "Monday" in f else ("Wednesday" if "Wednesday" in f else None)
    if tag is None:
        continue
    df = pd.read_csv(f, usecols=lambda c: c.strip() in FLAGS, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    raw[tag] = df
for tag, df in raw.items():
    print(f"\n{tag}: {len(df)} rows")
    for c in FLAGS:
        print(f"  {c:16s} max={df[c].max():>6}  top values={df[c].value_counts().head(5).to_dict()}")
    print("  PSH == ACK on %.1f%% of rows" % (100 * (df['PSH Flag Count'] == df['ACK Flag Count']).mean()))
mon = raw.get("Monday")
base = {KEY[c]: float((mon[c] > 0).mean()) for c in FLAGS} if mon is not None else {}

print("\n=== 2. Stored live flows, re-scored read-only ===")
db = MongoClient("mongodb://localhost:27017")["cyberdefense"]
q = {"source": "live_capture", "timestamp": {"$type": ["double", "int", "long"]}}
docs = [d for d in db["flows"].find(q, {"_id": 0, "features": 1})
        if d.get("features") and all(k in d["features"] for k in feats)]
X = np.array([[d["features"][k] for k in feats] for d in docs], dtype=float)
print("live flows:", len(X))
print("\nP(flag present): live vs CIC Monday (benign only)")
for name in KEY.values():
    i = feats.index(name)
    print("  %-10s live %6.2f%%   CIC-benign %6.2f%%   live P(count>1) %6.2f%%   live max %g" % (
        name, 100 * (X[:, i] > 0).mean(), 100 * base.get(name, float('nan')), 100 * (X[:, i] > 1).mean(), X[:, i].max()))


def unsup(Xm):
    Z = scaler.transform(np.log1p(np.clip(Xm, 0, None)))
    s = iso.score_samples(Z)
    e = ((ae.predict(Z) - Z) ** 2).mean(axis=1)
    return s < iso_t, e > ae_t, Z


def report(label, Xm):
    i_f, a_f, Z = unsup(Xm)
    both = i_f & a_f
    print("  %-48s IF %5.2f%%  AE %5.2f%%  both %5.2f%% (%d flows)" % (
        label, 100 * i_f.mean(), 100 * a_f.mean(), 100 * both.mean(), both.sum()))
    return both, Z


fl = [feats.index(v) for v in KEY.values()]
Xb = X.copy()
Xb[:, fl] = (Xb[:, fl] > 0).astype(float)
Xn = Xb.copy()
Xn[:, feats.index("rst_count")] = 0.0
Xn[:, feats.index("fin_count")] = 0.0

print("\nflag rates by variant (design target for 'both': about 1%)")
report("A. as captured (flag counts)", X)
b1, Z1 = report("B. flags as 0/1 presence", Xb)
report("C. B with RST and FIN zeroed (diagnostic only)", Xn)
print("\nvariant B, feature with the largest |z| in each still-flagged flow:")
print(" ", dict(Counter(feats[int(np.argmax(np.abs(Z1[i])))] for i in np.where(b1)[0]).most_common()))

Pa = xg.predict_proba(pd.DataFrame(X, columns=feats))
Pb = xg.predict_proba(pd.DataFrame(Xb, columns=feats))
print("\nsupervised, top class != BENIGN: as captured %d | 0/1 flags %d | argmax changed on %d flows" % (
    (Pa.argmax(1) != benign).sum(), (Pb.argmax(1) != benign).sum(), (Pa.argmax(1) != Pb.argmax(1)).sum()))
