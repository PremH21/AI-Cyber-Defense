import sys
sys.path.insert(0, "api")
from collections import Counter
from pymongo import MongoClient
from engine import DefenseEngine

eng = DefenseEngine()
db = MongoClient("mongodb://localhost:27017")["cyberdefense"]
docs = list(db["flows"].find({"source": "live_capture", "features": {"$exists": True},
                              "timestamp": {"$type": ["double", "int", "long"]}}, {"_id": 0}))
c = Counter()
for d in docs:
    if d["protocol"] not in ("TCP", "UDP"):
        c["skipped (not TCP/UDP)"] += 1
        continue
    fv = dict(d["features"])
    for k in ("syn_count", "fin_count", "rst_count", "psh_count", "ack_count"):
        fv[k] = min(1, fv[k])
    f = {"src_ip": d["src_ip"], "dst_ip": d["dst_ip"], "src_port": d["src_port"], "dst_port": d["dst_port"],
         "protocol": d["protocol"], "ip_version": d.get("ip_version", "v4"), "start_ts": d["timestamp"],
         "duration_sec": fv["duration_us"] / 1e6, "fwd_packets": fv["fwd_packets"], "bwd_packets": fv["bwd_packets"],
         "fwd_bytes": fv["fwd_bytes"], "bwd_bytes": fv["bwd_bytes"]}
    c[eng.analyze_flow(f, features_override=fv, use_tracker=False)["verdict"]] += 1
n = sum(v for k, v in c.items() if not k.startswith("skipped"))
print("flows rescored:", n, dict(c))
print("incident rate: %.2f%%" % (100 * sum(v for k, v in c.items() if k in ("known_attack", "suspected_zero_day")) / max(n, 1)))
