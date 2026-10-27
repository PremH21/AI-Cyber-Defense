import sys
from collections import Counter
import numpy as np
from pymongo import MongoClient

db = MongoClient("mongodb://localhost:27017")["cyberdefense"]
col = db["flows"]
print("flows total:", col.count_documents({}))
print("by source:", Counter(d.get("source") for d in col.find({}, {"source": 1})))

q = {"timestamp": {"$type": ["double", "int", "long"]}, "source": "live_capture"}
docs = list(col.find(q, {"_id": 0}))
if not docs:
    sys.exit("no new-engine live_capture flows found")

n = len(docs)
ts = [d["timestamp"] for d in docs]
print("\nnew-engine live flows:", n, " window: %.1f minutes" % ((max(ts) - min(ts)) / 60))
flag = [d for d in docs if d.get("predicted_label") == "attack"]
print("flagged:", len(flag), " rate: %.4f" % (len(flag) / n))
print("by verdict:", Counter(d.get("verdict") for d in flag))
print("by attack_type:", Counter(d.get("attack_type") for d in flag))
print("by layers:", Counter(tuple(d.get("detection_layers", [])) for d in flag))
print("has features stored:", sum(1 for d in docs if d.get("features")), "of", n)


def u(d, k):
    return ((d.get("ml") or {}).get("unsupervised") or {}).get(k)


ae = np.array([u(d, "autoencoder_error") for d in docs if u(d, "autoencoder_error") is not None])
iso = np.array([u(d, "iforest_score") for d in docs if u(d, "iforest_score") is not None])
first = next(d for d in docs if u(d, "autoencoder_threshold") is not None)
ae_t, iso_t = u(first, "autoencoder_threshold"), u(first, "iforest_threshold")
print("\n--- unsupervised models on ALL flows (design target was 1% on CIC benign) ---")
print("autoencoder error percentiles 50/90/95/99:", np.percentile(ae, [50, 90, 95, 99]).round(3), " threshold", ae_t)
print("iforest score percentiles 1/5/10/50:", np.percentile(iso, [1, 5, 10, 50]).round(4), " threshold", iso_t)
print("autoencoder flags: %.3f  iforest flags: %.3f  both: %.3f" % (
    (ae > ae_t).mean(), (iso < iso_t).mean(),
    np.mean([(u(d, 'autoencoder_error') > ae_t) and (u(d, 'iforest_score') < iso_t)
             for d in docs if u(d, 'autoencoder_error') is not None])))
print("supervised known_attack verdicts:", sum(1 for d in flag if d.get("verdict") == "known_attack"))


def rate(label, pred):
    sub = [d for d in docs if pred(d)]
    if not sub:
        print(f"{label:48s} n=0")
        return
    f = sum(1 for d in sub if d.get("predicted_label") == "attack")
    print(f"{label:48s} n={len(sub):6d} flagged={f:5d} rate={f / len(sub):.3f}")


def rev(d):
    return (d.get("src_port") or 0) < 1024 and (d.get("dst_port") or 0) >= 1024


def syn(d):
    return (d.get("features") or {}).get("syn_count")


print("\n--- is the flagged rate tied to flow orientation or mid-connection pickup? ---")
rate("server-to-client oriented (src<1024, dst>=1024)", rev)
rate("client-to-server oriented (everything else)", lambda d: not rev(d))
rate("no SYN seen (syn_count == 0)", lambda d: syn(d) == 0)
rate("SYN seen (syn_count > 0)", lambda d: (syn(d) or 0) > 0)
print("\n--- by protocol and IP version ---")
for key in sorted({(d.get("protocol"), d.get("ip_version")) for d in docs}, key=str):
    rate(str(key), lambda d, key=key: (d.get("protocol"), d.get("ip_version")) == key)
print("\ntop flagged destinations:")
for k, v in Counter((d.get("dst_ip"), d.get("dst_port")) for d in flag).most_common(8):
    print("  ", k, v)
