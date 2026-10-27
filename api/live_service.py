"""
Live detection service.

  uvicorn api.live_service:app --port 8000

Endpoints
  POST /ingest_live     flow from the capture agent  -> full detection record
  POST /ingest_event    non-flow event (ARP spoof)    -> detection record
  POST /honeypot_alert  decoy-file access             -> critical alert
  GET  /incidents       recent ALERTS (non-benign)
  GET  /flows           recent flows incl. benign
  GET  /stats           counters and latency
  GET  /health

Storage: MongoDB if reachable (MONGO_URI), otherwise an in-memory buffer, so the
service still runs with Docker off.
"""
import os
import threading
import time
import uuid
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from api.engine import DefenseEngine, band
except ImportError:  # run directly from inside api/
    from engine import DefenseEngine, band

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "cyberdefense"
STATE = {"engine": None, "db": None, "lock": threading.Lock()}
MEM_FLOWS = deque(maxlen=3000)
MEM_ALERTS = deque(maxlen=1000)
COUNTERS = {"flows": 0, "alerts": 0, "flagged_flows": 0, "latency_sum": 0.0, "started": time.time()}
ACTIVE = {}          # (src, dst, attack_type) -> open incident record
INCIDENT_WINDOW = 30  # seconds of silence before the same detection opens a new incident
BY_TYPE, BY_BAND = Counter(), Counter()


@asynccontextmanager
async def lifespan(app):
    print("Loading detection engine...", flush=True)
    STATE["engine"] = DefenseEngine()
    STATE["engine"].warmup()
    print(f"Engine ready: {len(STATE['engine'].classes)} classes, {len(STATE['engine'].features)} features, "
          f"{STATE['engine'].rl_states} RL states", flush=True)
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client.server_info()
        db = client[DB_NAME]
        db["flows"].create_index("created_at", expireAfterSeconds=6 * 3600)
        STATE["db"] = db
        print(f"Storage: MongoDB at {MONGO_URI}", flush=True)
    except Exception as e:
        print(f"Storage: in-memory (MongoDB unavailable: {type(e).__name__})", flush=True)
    yield


app = FastAPI(title="AI-Driven Autonomous Defense Framework - live service", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class FlowIn(BaseModel):
    src_ip: str
    dst_ip: str
    src_port: int = 0
    dst_port: int = 0
    protocol: str = "TCP"
    ip_version: str = "v4"
    start_ts: float
    duration_sec: float
    fwd_packets: int
    bwd_packets: int
    fwd_bytes: float
    bwd_bytes: float
    syn_count: int = 0
    fin_count: int = 0
    rst_count: int = 0
    psh_count: int = 0
    ack_count: int = 0


class EventIn(BaseModel):
    kind: str
    ip: str
    old_mac: str
    new_mac: str


class HoneypotIn(BaseModel):
    file: str
    event: str = "accessed"
    detail: str = ""


def _store(rec):
    COUNTERS["flows"] += 1
    COUNTERS["latency_sum"] += rec.get("latency_ms", 0)
    is_alert = rec["predicted_label"] == "attack"
    db = STATE["db"]
    now = rec["timestamp"]
    open_inc = None
    if is_alert:
        COUNTERS["flagged_flows"] += 1
        key = (rec["src_ip"], rec["dst_ip"], rec["attack_type"])
        open_inc = ACTIVE.get(key)
        if open_inc is not None and now - open_inc["last_seen"] <= INCIDENT_WINDOW:
            open_inc["repeat_count"] += 1
            open_inc["last_seen"] = now
            rec["incident_id"] = open_inc["incident_id"]
            if db is not None:
                try:
                    db["incidents"].update_one({"incident_id": open_inc["incident_id"]},
                                               {"$inc": {"repeat_count": 1}, "$set": {"last_seen": now}})
                except Exception as e:
                    print("Mongo update failed:", e, flush=True)
            is_new_incident = False
        else:
            rec["incident_id"] = uuid.uuid4().hex[:8]
            rec["repeat_count"] = 1
            rec["last_seen"] = now
            ACTIVE[key] = rec
            is_new_incident = True
    if db is not None:
        try:
            doc = {**rec, "created_at": datetime.now(timezone.utc)}
            db["flows"].insert_one(dict(doc))
            if is_alert and is_new_incident:
                db["incidents"].insert_one(dict(doc))
        except Exception as e:
            print("Mongo write failed:", e, flush=True)
    MEM_FLOWS.append(rec)
    if is_alert and is_new_incident:
        COUNTERS["alerts"] += 1
        BY_TYPE[rec["attack_type"]] += 1
        BY_BAND[rec["severity_band"]] += 1
        MEM_ALERTS.append(rec)


def _clean(d):
    d.pop("_id", None)
    return d


@app.get("/health")
def health():
    return {"status": "ok", "engine_loaded": STATE["engine"] is not None,
            "rl_policy_states": STATE["engine"].rl_states if STATE["engine"] else 0,
            "storage": "mongodb" if STATE["db"] is not None else "memory"}


@app.post("/ingest_live")
def ingest_live(flow: FlowIn):
    with STATE["lock"]:
        rec = STATE["engine"].analyze_flow(flow.model_dump())
    _store(rec)
    return rec


@app.post("/ingest_event")
def ingest_event(ev: EventIn):
    try:
        rec = STATE["engine"].analyze_event(ev.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    _store(rec)
    return rec


@app.post("/honeypot_alert")
def honeypot_alert(h: HoneypotIn):
    eng = STATE["engine"]
    sev = 10
    action, src = eng.choose_action("malware_backdoor", sev, "high")
    rec = {
        "timestamp": time.time(), "source": "honeypot", "event_type": "honeypot",
        "src_ip": "local-host", "dst_ip": "local-host", "src_port": 0, "dst_port": 0, "protocol": "FILE",
        "verdict": "behavioural_detection", "predicted_label": "attack",
        "attack_type": "Honeypot Decoy Access (possible ransomware / intruder)",
        "category": "Intrusion", "slide_category": "Ransomware", "threat_type": "malware_backdoor",
        "confidence": 1.0, "severity": sev, "severity_band": band(sev), "asset_criticality": "high",
        "action_taken": action, "action_source": src, "detection_layers": ["honeypot"], "ml": {"applied": False},
        "explanation": {"type": "behavioural_evidence", "evidence": {"file": h.file, "event": h.event, "detail": h.detail}},
        "why": f"Decoy file {h.file} was {h.event}. Decoys have no legitimate use, so any access is hostile (zero false positives).",
        "latency_ms": 0.0,
    }
    _store(rec)
    return rec


@app.get("/incidents")
def incidents(limit: int = 100):
    db = STATE["db"]
    if db is not None:
        try:
            docs = [_clean(d) for d in db["incidents"].find({}, {"features": 0}).sort("timestamp", -1).limit(limit)]
            for d in docs:
                d.pop("created_at", None)
            return {"count": len(docs), "incidents": docs, "storage": "mongodb"}
        except Exception:
            pass
    docs = [{k: v for k, v in r.items() if k != "features"} for r in list(MEM_ALERTS)[-limit:]][::-1]
    return {"count": len(docs), "incidents": docs, "storage": "memory"}


@app.get("/flows")
def flows(limit: int = 100):
    docs = [{k: v for k, v in r.items() if k not in ("features", "explanation")}
            for r in list(MEM_FLOWS)[-limit:]][::-1]
    return {"count": len(docs), "flows": docs}


@app.get("/stats")
def stats():
    n = COUNTERS["flows"]
    return {
        "total_flows_analyzed": n, "total_incidents": COUNTERS["alerts"], "flows_flagged": COUNTERS["flagged_flows"],
        "flagged_flow_rate": round(COUNTERS["flagged_flows"] / n, 4) if n else None,
        "avg_detection_latency_ms": round(COUNTERS["latency_sum"] / n, 3) if n else None,
        "alerts_by_attack_type": dict(BY_TYPE), "alerts_by_severity_band": dict(BY_BAND),
        "uptime_seconds": int(time.time() - COUNTERS["started"]),
        "storage": "mongodb" if STATE["db"] is not None else "memory",
    }
