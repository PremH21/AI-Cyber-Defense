"""
Defense engine - the real-time decision core of the framework.

For every network flow (or event) it runs, in order:
  READ        build the 16 flow features
  UNDERSTAND  (1) supervised XGBoost       -> known attack type + confidence
              (2) Isolation Forest + Deep Autoencoder (trained on benign only)
                                            -> anomaly flags for NEW attacks
              (3) behavioural layer        -> scans, brute force, floods,
                                              lateral movement, beaconing, exfiltration
  SCORE       severity 1-10 from a documented formula
  FLAG        verdict: known_attack | behavioural_detection | suspected_zero_day | benign
  ALERT       explanation (SHAP / evidence / reconstruction error) + plain-English reason
  PROTECT     RL policy picks the response action

Nothing here is scripted: every field is computed from the flow it is given.
"""
import ipaddress
import json
import os
import time
from collections import defaultdict, deque

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.environ.get("MODEL_DIR", "ml-engine/models")
QTABLE_PATH = os.environ.get("QTABLE_PATH", "response-engine/models/q_table_threat.json")

# --------------------------------------------------------------------------
# Taxonomy: (attack_type, category, slide_category, rl_threat_type, base_risk 1-10)
# --------------------------------------------------------------------------
TAXONOMY = {
    "BENIGN": ("Benign", "Benign", "N/A", "benign", 0),
    "Bot": ("Botnet Activity", "Botnet / C2", "C2 Communication", "botnet", 8),
    "DDoS": ("DDoS", "Denial of Service", "DDoS", "dos_ddos", 8),
    "DoS GoldenEye": ("DoS GoldenEye", "Denial of Service", "DDoS", "dos_ddos", 6),
    "DoS Hulk": ("DoS Hulk", "Denial of Service", "DDoS", "dos_ddos", 6),
    "DoS Slowhttptest": ("DoS Slowhttptest", "Denial of Service", "DDoS", "dos_ddos", 6),
    "DoS slowloris": ("DoS Slowloris", "Denial of Service", "DDoS", "dos_ddos", 6),
    "FTP-Patator": ("FTP Brute Force", "Credential Attack", "Insider Threats (credential abuse)", "credential_attack", 5),
    "SSH-Patator": ("SSH Brute Force", "Credential Attack", "Insider Threats (credential abuse)", "credential_attack", 5),
    "Heartbleed": ("Heartbleed Exploit", "Exploit", "Zero-Day", "malware_backdoor", 9),
    "Infiltration": ("Network Infiltration", "Infiltration", "APT", "infiltration", 9),
    "PortScan": ("Port Scan", "Reconnaissance", "Lateral Movement (recon)", "recon_scan", 3),
    "Web Attack": ("Web Application Attack", "Web Attack", "Phishing / web vector", "web_attack", 6),
}
# Classes with poor precision at flow level must be very confident before alerting.
MIN_CONF = {"Bot": 0.95, "Web Attack": 0.95, "Infiltration": 0.90}
DEFAULT_MIN_CONF = 0.60

BEHAVIOUR_TAXONOMY = {
    "port_scan": ("Port Scan", "Reconnaissance", "Lateral Movement (recon)", "recon_scan", 3),
    "network_sweep": ("Network Sweep", "Reconnaissance", "Lateral Movement (recon)", "recon_scan", 3),
    "brute_force": ("Credential Brute Force", "Credential Attack", "Insider Threats (credential abuse)", "credential_attack", 5),
    "syn_flood": ("SYN Flood", "Denial of Service", "DDoS", "dos_ddos", 7),
    "ddos": ("Distributed Flood", "Denial of Service", "DDoS", "dos_ddos", 8),
    "conn_flood": ("Connection Flood", "Denial of Service", "DDoS", "dos_ddos", 6),
    "lateral_movement": ("Lateral Movement", "Lateral Movement", "Lateral Movement", "infiltration", 7),
    "beaconing": ("Suspected C2 Beaconing", "Botnet / C2", "C2 Communication", "botnet", 5),
    "exfiltration": ("Possible Data Exfiltration", "Data Exfiltration", "APT", "infiltration", 8),
    "arp_spoof": ("ARP Spoofing", "Man-in-the-Middle", "MITM", "infiltration", 8),
    "zero_day": ("Unknown Anomalous Behaviour", "Suspected Zero-Day", "Zero-Day", "malware_backdoor", 6),
}

ASSET_MULT = {"low": 0.8, "medium": 1.0, "high": 1.2}
CRITICAL_PORTS = {22, 445, 1433, 3306, 3389, 5432, 6379, 27017, 5985}

# Behavioural thresholds (window seconds in the rule names). Tunable; validated
# by launching real attacks in the lab (nmap / hydra / hping3) against own devices.
PORT_SCAN_PORTS, PORT_SCAN_WIN = 15, 10
SWEEP_HOSTS, SWEEP_WIN = 20, 10
AUTH_PORTS = {21, 22, 23, 110, 143, 445, 1433, 3306, 3389, 5432, 5900, 5985}
BRUTE_COUNT, BRUTE_WIN = 8, 30
FLOOD_WIN, FLOOD_WEB, FLOOD_OTHER = 10, 200, 60
WEB_PORTS = {80, 443, 8080, 8443}
SYN_FLOOD_COUNT, SYN_FLOOD_WIN = 100, 5
DDOS_COUNT, DDOS_SOURCES, DDOS_WIN = 300, 20, 5
LATERAL_PORTS = {445, 3389, 22, 5985, 135, 139}
LATERAL_HOSTS, LATERAL_WIN = 5, 60
BEACON_MIN_N, BEACON_MAX_CV, BEACON_MIN_SPAN = 10, 0.10, 60
EXFIL_BYTES, EXFIL_WIN = 100e6, 120
IGNORED_PORTS = {53, 67, 68, 123, 137, 138, 1900, 5353, 5355}
COOLDOWN = 60


def _ip(s):
    try:
        return ipaddress.ip_address(s)
    except ValueError:
        return None


def is_private(s):
    a = _ip(s)
    return bool(a and (a.is_private or a.is_link_local) and not a.is_loopback)


def is_ignorable(s):
    a = _ip(s)
    return bool(a and (a.is_multicast or a.is_loopback or a.is_unspecified or str(a) == "255.255.255.255"))


def band(sev):
    return "LOW" if sev <= 3 else "MEDIUM" if sev <= 6 else "HIGH" if sev <= 8 else "CRITICAL"


# --------------------------------------------------------------------------
# Behavioural layer: needs no training data, works on any network.
# --------------------------------------------------------------------------
class BehaviourTracker:
    def __init__(self):
        self.by_src = defaultdict(lambda: deque(maxlen=6000))
        self.by_dst = defaultdict(lambda: deque(maxlen=6000))
        self.starts = defaultdict(lambda: deque(maxlen=200))
        self.cool = {}
        self.flagged_src = {}

    def _fire(self, key, now):
        if now - self.cool.get(key, -1e9) < COOLDOWN:
            return False
        self.cool[key] = now
        return True

    @staticmethod
    def _conf(observed, threshold):
        return float(min(1.0, 0.5 + 0.25 * observed / max(threshold, 1e-9)))

    def update(self, f):
        src, dst, dport = f["src_ip"], f["dst_ip"], int(f["dst_port"])
        if is_ignorable(dst) or is_ignorable(src):
            return []
        now = float(f.get("start_ts") or time.time())
        half = f["bwd_packets"] == 0
        proto = f.get("protocol", "TCP")
        out = []

        dq = self.by_src[src]
        dq.append((now, dst, dport, half, f["fwd_bytes"]))
        while dq and now - dq[0][0] > 150:
            dq.popleft()
        dd = self.by_dst[(dst, dport)]
        dd.append((now, src, half))
        while dd and now - dd[0][0] > 30:
            dd.popleft()

        def det(kind, observed, threshold, window, evidence, target=dst):
            if self._fire((kind, src, target), now):
                self.flagged_src[src] = now
                out.append({"type": kind, "observed": observed, "threshold": threshold,
                            "window_s": window, "confidence": self._conf(observed, threshold),
                            "evidence": evidence, "src_ip": src, "target": target})

        if dport not in IGNORED_PORTS and proto in ("TCP", "UDP"):
            # port scan: many distinct ports on one host
            ports = {e[2] for e in dq if now - e[0] <= PORT_SCAN_WIN and e[1] == dst}
            if len(ports) >= PORT_SCAN_PORTS:
                det("port_scan", len(ports), PORT_SCAN_PORTS, PORT_SCAN_WIN,
                    {"distinct_ports": len(ports), "sample_ports": sorted(ports)[:12], "target_host": dst})

            # network sweep: same port on many hosts
            hosts = {e[1] for e in dq if now - e[0] <= SWEEP_WIN and e[2] == dport}
            if len(hosts) >= SWEEP_HOSTS:
                det("network_sweep", len(hosts), SWEEP_HOSTS, SWEEP_WIN,
                    {"distinct_hosts": len(hosts), "port": dport}, target=f"port:{dport}")

            # lateral movement: internal host reaching many internal hosts on admin ports
            if dport in LATERAL_PORTS and is_private(src) and is_private(dst):
                lat = {e[1] for e in dq if now - e[0] <= LATERAL_WIN and e[2] in LATERAL_PORTS and is_private(e[1])}
                if len(lat) >= LATERAL_HOSTS:
                    det("lateral_movement", len(lat), LATERAL_HOSTS, LATERAL_WIN,
                        {"internal_hosts_contacted": len(lat), "admin_ports": sorted(LATERAL_PORTS & {e[2] for e in dq})},
                        target="internal-hosts")

            # brute force: repeated connections to one auth service
            if dport in AUTH_PORTS:
                n = sum(1 for e in dq if now - e[0] <= BRUTE_WIN and e[1] == dst and e[2] == dport)
                if n >= BRUTE_COUNT:
                    det("brute_force", n, BRUTE_COUNT, BRUTE_WIN,
                        {"connections": n, "service_port": dport, "target_host": dst})

            # connection flood to one service
            thr = FLOOD_WEB if dport in WEB_PORTS else FLOOD_OTHER
            n = sum(1 for e in dq if now - e[0] <= FLOOD_WIN and e[1] == dst and e[2] == dport)
            if n >= thr and dport not in AUTH_PORTS:
                det("conn_flood", n, thr, FLOOD_WIN, {"connections": n, "service_port": dport, "target_host": dst})

            # SYN flood: many half-open connections
            if proto == "TCP":
                nh = sum(1 for e in dq if now - e[0] <= SYN_FLOOD_WIN and e[1] == dst and e[2] == dport and e[3])
                if nh >= SYN_FLOOD_COUNT:
                    det("syn_flood", nh, SYN_FLOOD_COUNT, SYN_FLOOD_WIN,
                        {"half_open_connections": nh, "service_port": dport, "target_host": dst})

            # distributed flood (destination-centric)
            recent = [e for e in dd if now - e[0] <= DDOS_WIN]
            if len(recent) >= DDOS_COUNT and len({e[1] for e in recent}) >= DDOS_SOURCES:
                key = ("ddos", dst, dport)
                if self._fire(key, now):
                    out.append({"type": "ddos", "observed": len(recent), "threshold": DDOS_COUNT,
                                "window_s": DDOS_WIN, "confidence": self._conf(len(recent), DDOS_COUNT),
                                "evidence": {"flows": len(recent), "distinct_sources": len({e[1] for e in recent}),
                                             "victim": f"{dst}:{dport}"},
                                "src_ip": "multiple", "target": dst})

        # exfiltration: large volume from internal host to one external destination
        if is_private(src) and not is_private(dst):
            vol = sum(e[4] for e in dq if now - e[0] <= EXFIL_WIN and e[1] == dst)
            if vol >= EXFIL_BYTES:
                det("exfiltration", vol, EXFIL_BYTES, EXFIL_WIN, {"bytes_sent": int(vol), "external_host": dst})

        # C2 beaconing: very regular connections to one external host
        if not is_private(dst) and dport not in IGNORED_PORTS:
            st = self.starts[(src, dst, dport)]
            st.append(now)
            if len(st) >= BEACON_MIN_N:
                ts = np.array(st)
                iv = np.diff(ts)
                if ts[-1] - ts[0] >= BEACON_MIN_SPAN and iv.mean() > 0:
                    cv = float(iv.std() / iv.mean())
                    if cv < BEACON_MAX_CV and 5 <= iv.mean() <= 300:
                        det("beaconing", len(ts), BEACON_MIN_N, int(ts[-1] - ts[0]),
                            {"connections": len(ts), "mean_interval_s": round(float(iv.mean()), 2),
                             "interval_cv": round(cv, 4), "destination": f"{dst}:{dport}"},
                            target=f"{dst}:{dport}")
        return out


# --------------------------------------------------------------------------
class DefenseEngine:
    def __init__(self, model_dir=MODEL_DIR, qtable_path=QTABLE_PATH):
        need = ["live_multiclass_cicids.joblib", "live_multiclass_meta.json", "live_iforest.joblib",
                "live_autoencoder.joblib", "live_unsup_scaler.joblib", "live_unsup_meta.json"]
        missing = [n for n in need if not os.path.exists(f"{model_dir}/{n}")]
        if missing:
            raise FileNotFoundError(f"Missing trained models in {model_dir}: {missing}. "
                                    "Run ml-engine/train_live_multiclass.py and ml-engine/zero_day_eval.py first.")
        meta = json.load(open(f"{model_dir}/live_multiclass_meta.json"))
        self.classes, self.features = meta["classes"], meta["features"]
        self.benign_id = self.classes.index("BENIGN")
        self.xgb = joblib.load(f"{model_dir}/live_multiclass_cicids.joblib")
        um = json.load(open(f"{model_dir}/live_unsup_meta.json"))
        if um["features"] != self.features:
            raise ValueError("Supervised and unsupervised models were trained on different feature sets.")
        self.iso_thr, self.ae_thr = um["iso_threshold"], um["ae_threshold"]
        self.iso = joblib.load(f"{model_dir}/live_iforest.joblib")
        self.ae = joblib.load(f"{model_dir}/live_autoencoder.joblib")
        self.scaler = joblib.load(f"{model_dir}/live_unsup_scaler.joblib")
        self.tracker = BehaviourTracker()
        self.explainer = None
        self.q = {}
        if os.path.exists(qtable_path):
            raw = json.load(open(qtable_path))
            # valid keys look like "threat_type|severity|asset" mapping to {action: q_value}
            self.q = {k: v for k, v in raw.items() if k.count("|") == 2 and isinstance(v, dict)}
        self.rl_states = len(self.q)
        if self.rl_states == 0:
            print(f"WARNING: no usable threat-aware RL Q-table at {qtable_path}. Actions will come from the "
                  "fallback rule, NOT the RL policy. Run response-engine/rl_response_agent.py.", flush=True)
        else:
            print(f"RL policy loaded: {self.rl_states} states from {qtable_path}", flush=True)
        self.own_ips = {x for x in os.environ.get("OWN_IPS", "").split(",") if x}

    def warmup(self):
        """Load SHAP and JIT paths once so the first real alert is not slow."""
        f = {"src_ip": "10.255.0.1", "dst_ip": "10.255.0.2", "src_port": 1, "dst_port": 80, "protocol": "TCP",
             "start_ts": 1.0, "duration_sec": 0.5, "fwd_packets": 5, "bwd_packets": 4, "fwd_bytes": 300,
             "bwd_bytes": 900, "syn_count": 1, "ack_count": 5}
        fv = self.build_features(f)
        X, probs = self.supervised(fv)
        self.unsupervised(fv)
        self.shap_top5(X, int(probs.argmax()))

    # ---- READ ------------------------------------------------------------
    def build_features(self, f):
        dur = max(float(f["duration_sec"]), 0.0)
        fp, bp, fb, bb = f["fwd_packets"], f["bwd_packets"], f["fwd_bytes"], f["bwd_bytes"]
        tp, tb = fp + bp, fb + bb
        d = {
            "duration_us": dur * 1e6, "fwd_packets": fp, "bwd_packets": bp, "fwd_bytes": fb, "bwd_bytes": bb,
            "fwd_pkt_len_mean": fb / fp if fp else 0.0, "bwd_pkt_len_mean": bb / bp if bp else 0.0,
            "avg_packet_size": tb / tp if tp else 0.0,
            "bytes_per_sec": tb / dur if dur > 0 else 0.0, "packets_per_sec": tp / dur if dur > 0 else 0.0,
            "dst_port": f["dst_port"], "syn_count": f.get("syn_count", 0), "fin_count": f.get("fin_count", 0),
            "rst_count": f.get("rst_count", 0), "psh_count": f.get("psh_count", 0), "ack_count": f.get("ack_count", 0),
        }
        return {k: float(d[k]) for k in self.features}

    # ---- UNDERSTAND ------------------------------------------------------
    def supervised(self, fv):
        X = pd.DataFrame([fv], columns=self.features)
        p = self.xgb.predict_proba(X)[0]
        return X, p

    def unsupervised(self, fv):
        x = np.array([[fv[k] for k in self.features]], dtype=float)
        Z = self.scaler.transform(np.log1p(np.clip(x, 0, None)))
        iso_score = float(self.iso.score_samples(Z)[0])
        err_vec = (self.ae.predict(Z) - Z)[0] ** 2
        ae_err = float(err_vec.mean())
        return {"iforest_score": round(iso_score, 4), "iforest_flag": bool(iso_score < self.iso_thr),
                "autoencoder_error": round(ae_err, 4), "autoencoder_flag": bool(ae_err > self.ae_thr),
                "iforest_threshold": round(self.iso_thr, 4), "autoencoder_threshold": round(self.ae_thr, 4)}, err_vec

    def shap_top5(self, X, class_idx):
        if self.explainer is None:
            import shap
            self.explainer = shap.TreeExplainer(self.xgb)
        sv = self.explainer.shap_values(X)
        if isinstance(sv, list):
            v = np.asarray(sv[class_idx])[0]
        else:
            sv = np.asarray(sv)
            v = sv[0, :, class_idx] if sv.ndim == 3 else sv[0]
        idx = np.argsort(np.abs(v))[::-1][:5]
        return [{"feature": self.features[i], "value": round(float(X.iloc[0, i]), 4),
                 "shap_contribution": round(float(v[i]), 4),
                 "direction": "pushes toward this attack" if v[i] > 0 else "pushes away from this attack"}
                for i in idx]

    # ---- SCORE / PROTECT -------------------------------------------------
    @staticmethod
    def severity(base, conf, asset, corroborated):
        s = round(base * (0.5 + 0.5 * conf) * ASSET_MULT[asset]) + (1 if corroborated else 0)
        return int(min(10, max(1, s)))

    def asset_level(self, dst_ip, dst_port):
        if dst_ip in self.own_ips or is_private(dst_ip):
            return "high" if int(dst_port) in CRITICAL_PORTS else "medium"
        return "low"

    def choose_action(self, threat, sev, asset):
        if threat == "benign":
            return "allow", "n/a"
        for d in range(0, 4):
            for s in {sev - d, sev + d}:
                for a in (asset, "medium", "low", "high"):
                    q = self.q.get(f"{threat}|{s}|{a}")
                    if isinstance(q, dict) and q:
                        return max(q, key=q.get), "rl_policy"
        return ("alert_only" if sev <= 3 else "quarantine" if sev <= 6 else "block"), "fallback_rule"

    # ---- main pipeline ---------------------------------------------------
    def analyze_flow(self, f, features_override=None, use_tracker=True):
        t0 = time.perf_counter()
        fv = features_override or self.build_features(f)
        n_pkts = f["fwd_packets"] + f["bwd_packets"]
        ml_ok = features_override is not None or (n_pkts >= 2 and f["duration_sec"] > 0)

        cands, ml = [], {"applied": ml_ok}
        X = err_vec = None
        if ml_ok:
            X, probs = self.supervised(fv)
            top = int(probs.argmax())
            cls = self.classes[top]
            ml.update({"top_class": cls, "top_probability": round(float(probs[top]), 4),
                       "attack_probability": round(float(1 - probs[self.benign_id]), 4)})
            unsup, err_vec = self.unsupervised(fv)
            ml["unsupervised"] = unsup
            if top != self.benign_id and probs[top] >= MIN_CONF.get(cls, DEFAULT_MIN_CONF):
                t = TAXONOMY.get(cls, (cls, "Unknown", "Zero-Day", "malware_backdoor", 5))
                cands.append({"layer": "supervised_xgboost", "kind": "known_attack", "tax": t,
                              "conf": float(probs[top]), "class_idx": top})

        dets = self.tracker.update(f) if use_tracker else []
        for d in dets:
            cands.append({"layer": f"behavioural:{d['type']}", "kind": "behavioural_detection",
                          "tax": BEHAVIOUR_TAXONOMY[d["type"]], "conf": d["confidence"], "det": d})

        # Both unsupervised models agreeing is a zero-day suspicion only when nothing
        # more specific explains the flow; otherwise it corroborates the specific finding.
        both_unsup = bool(ml_ok and ml["unsupervised"]["iforest_flag"] and ml["unsupervised"]["autoencoder_flag"])
        corroborated_by_unsup = False
        if both_unsup:
            if cands:
                corroborated_by_unsup = True
            else:
                cands.append({"layer": "isolation_forest+autoencoder", "kind": "suspected_zero_day",
                              "tax": BEHAVIOUR_TAXONOMY["zero_day"], "conf": 0.7})

        asset = self.asset_level(f["dst_ip"], f["dst_port"])
        rec = self._base_record(f, fv, ml, asset)

        if cands:
            for c in cands:
                c["potential"] = c["tax"][4] * (0.5 + 0.5 * c["conf"])
            cands.sort(key=lambda c: -c["potential"])
            p = cands[0]
            attack_type, category, slide, threat, base = p["tax"]
            layers = sorted({c["layer"] for c in cands} | ({"isolation_forest+autoencoder"} if corroborated_by_unsup else set()))
            sev = self.severity(base, p["conf"], asset, corroborated=len(layers) > 1)
            action, src = self.choose_action(threat, sev, asset)
            rec.update({"verdict": p["kind"], "predicted_label": "attack", "attack_type": attack_type,
                        "category": category, "slide_category": slide, "threat_type": threat,
                        "confidence": round(p["conf"], 4), "severity": sev, "severity_band": band(sev),
                        "action_taken": action, "action_source": src, "detection_layers": layers,
                        "also_detected": [c["tax"][0] for c in cands[1:]]})
            rec["explanation"], rec["why"] = self._explain(p, X, err_vec, attack_type, fv)
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return rec

    def analyze_event(self, ev):
        """Non-flow events, e.g. ARP spoofing seen by the agent."""
        kind = ev.get("kind")
        if kind != "arp_spoof":
            raise ValueError(f"unknown event kind {kind}")
        t0 = time.perf_counter()
        t = BEHAVIOUR_TAXONOMY["arp_spoof"]
        sev = self.severity(t[4], 0.9, "high", False)
        action, src = self.choose_action(t[3], sev, "high")
        ev_data = {"ip": ev["ip"], "previous_mac": ev["old_mac"], "new_mac": ev["new_mac"]}
        return {
            "timestamp": time.time(), "source": "live_capture", "event_type": "arp",
            "src_ip": ev["ip"], "dst_ip": "LAN", "src_port": 0, "dst_port": 0, "protocol": "ARP",
            "verdict": "behavioural_detection", "predicted_label": "attack", "attack_type": t[0],
            "category": t[1], "slide_category": t[2], "threat_type": t[3], "confidence": 0.9,
            "severity": sev, "severity_band": band(sev), "asset_criticality": "high",
            "action_taken": action, "action_source": src, "detection_layers": ["behavioural:arp_spoof"],
            "ml": {"applied": False},
            "explanation": {"type": "behavioural_evidence", "evidence": ev_data},
            "why": (f"IP {ev['ip']} was bound to MAC {ev['old_mac']} and is now claimed by {ev['new_mac']}. "
                    "A changing IP-to-MAC binding is the signature of ARP spoofing (man-in-the-middle)."),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    # ---- helpers ----------------------------------------------------------
    def _base_record(self, f, fv, ml, asset):
        return {
            "timestamp": time.time(), "source": "live_capture",
            "src_ip": f["src_ip"], "dst_ip": f["dst_ip"], "src_port": f["src_port"], "dst_port": f["dst_port"],
            "protocol": f.get("protocol", "TCP"), "ip_version": f.get("ip_version", "v4"),
            "features": {k: round(v, 4) for k, v in fv.items()},
            "ml": ml, "verdict": "benign", "predicted_label": "benign", "attack_type": "Benign",
            "category": "Benign", "slide_category": "N/A", "threat_type": "benign",
            "confidence": ml.get("top_probability") if ml.get("top_class") == "BENIGN" else None,
            "severity": 0, "severity_band": "NONE", "asset_criticality": asset,
            "action_taken": "allow", "action_source": "n/a", "detection_layers": [],
            "also_detected": [], "explanation": None,
            "why": "No supervised, unsupervised or behavioural layer flagged this flow.",
        }

    def _explain(self, p, X, err_vec, attack_type, fv):
        if p["layer"] == "supervised_xgboost":
            top5 = self.shap_top5(X, p["class_idx"])
            drivers = ", ".join(f"{t['feature']}={t['value']:g}" for t in top5[:3])
            return ({"type": "shap", "explained_class": attack_type, "top_5_features": top5},
                    f"Supervised XGBoost classified this flow as {attack_type} with {p['conf']:.0%} confidence. "
                    f"Strongest SHAP drivers: {drivers}.")
        if p["kind"] == "suspected_zero_day":
            idx = np.argsort(err_vec)[::-1][:5]
            top5 = [{"feature": self.features[i], "value": round(fv[self.features[i]], 4),
                     "reconstruction_error": round(float(err_vec[i]), 4)} for i in idx]
            return ({"type": "autoencoder_reconstruction_error", "top_5_features": top5},
                    "Matches no known attack class, but both unsupervised models (Isolation Forest and the "
                    "autoencoder, trained only on normal traffic) judge it abnormal. Least-normal features: "
                    + ", ".join(t["feature"] for t in top5[:3]) + ".")
        d = p["det"]
        ev = ", ".join(f"{k}={v}" for k, v in d["evidence"].items())
        return ({"type": "behavioural_evidence", "rule": d["type"], "observed": d["observed"],
                 "threshold": d["threshold"], "window_seconds": d["window_s"], "evidence": d["evidence"]},
                f"{attack_type} rule fired: observed {d['observed']:g} against a threshold of {d['threshold']:g} "
                f"within {d['window_s']}s ({ev}).")
