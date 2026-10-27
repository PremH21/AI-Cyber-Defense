"""
Offline sanity test of the defense engine. Needs no root, no network, no database.
It generates flow sequences that look like real attack tools and checks that the
engine flags them with the right type, severity and explanation.

Run from the project root:  python api/test_engine.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from engine import DefenseEngine  # noqa: E402

eng = DefenseEngine()
PASS, FAIL = [], []


def flow(src, dst, dport, t, fwd=3, bwd=3, fb=200, bb=400, dur=0.5, proto="TCP", syn=1, ack=1, **kw):
    f = dict(src_ip=src, dst_ip=dst, src_port=40000 + int(t * 7) % 20000, dst_port=dport, protocol=proto,
             ip_version="v4", start_ts=t, duration_sec=dur, fwd_packets=fwd, bwd_packets=bwd,
             fwd_bytes=fb, bwd_bytes=bb, syn_count=syn, fin_count=0, rst_count=0, psh_count=1, ack_count=ack)
    f.update(kw)
    return f


def run(name, flows, expect_type):
    recs = [eng.analyze_flow(f) for f in flows]
    wanted = expect_type if isinstance(expect_type, (set, tuple, list)) else {expect_type}
    hits = [r for r in recs if r["attack_type"] in wanted]
    ok = len(hits) >= 1
    (PASS if ok else FAIL).append(name)
    print(f"\n=== {name}: expect '{expect_type}' -> {'PASS' if ok else 'FAIL'} "
          f"({len(hits)} alert(s) from {len(recs)} flows)")
    if hits:
        r = hits[0]
        print(f"  severity {r['severity']}/10 {r['severity_band']} | action {r['action_taken']} ({r['action_source']}) "
              f"| layers {r['detection_layers']} | asset {r['asset_criticality']}")
        print(f"  why: {r['why']}")
    else:
        print("  got:", sorted({r['attack_type'] for r in recs}))


ATT, VIC, NET = "192.168.1.50", "192.168.1.10", "151.101.1.69"

# nmap-style scan: 1-2 packet flows to many ports in a few seconds
run("Port scan", [flow(ATT, VIC, 1000 + i, 100 + i * 0.1, fwd=1, bwd=1, fb=0, bb=0, dur=0.001, syn=1, ack=0)
                  for i in range(40)], "Port Scan")

# hydra-style SSH brute force
run("SSH brute force", [flow("192.168.1.60", VIC, 22, 200 + i * 1.5, fwd=12, bwd=10, fb=900, bb=1500, dur=1.2)
                        for i in range(14)],
    {"Credential Brute Force", "SSH Brute Force", "FTP Brute Force"})

# SYN flood: many half-open connections to one service
run("SYN flood", [flow("192.168.1.70", VIC, 80, 300 + i * 0.01, fwd=1, bwd=0, fb=0, bb=0, dur=0.0, syn=1, ack=0)
                  for i in range(160)], "SYN Flood")

# distributed flood: many sources, one victim
run("Distributed flood", [flow(f"45.33.32.{i % 60 + 1}", VIC, 80, 400 + i * 0.01, fwd=1, bwd=0, fb=0, bb=0,
                               dur=0.0, syn=1, ack=0) for i in range(320)], "Distributed Flood")

# C2 beaconing: one external host, every 30 s with tiny jitter
run("C2 beaconing", [flow("192.168.1.80", NET, 8443, 500 + i * 30 + (i % 2) * 0.2, fwd=4, bwd=3, fb=300, bb=250,
                          dur=0.4) for i in range(14)], "Suspected C2 Beaconing")

# lateral movement: internal host hitting many internal hosts on SMB
run("Lateral movement", [flow("192.168.1.90", f"192.168.1.{20 + i}", 445, 600 + i * 2, fwd=6, bwd=5, fb=500, bb=800,
                              dur=0.8) for i in range(8)], "Lateral Movement")

# exfiltration: >100 MB to one external host
run("Data exfiltration", [flow("192.168.1.95", "34.120.55.10", 443, 700 + i * 5, fwd=70000, bwd=9000, fb=60_000_000,
                               bb=500_000, dur=4.0) for i in range(3)], "Possible Data Exfiltration")

# network sweep: one port across many hosts
run("Network sweep", [flow("10.9.0.5", f"52.84.10.{i + 1}", 3389, 800 + i * 0.2, fwd=1, bwd=0, fb=0, bb=0,
                           dur=0.0, syn=1, ack=0) for i in range(30)], "Network Sweep")

# ARP spoofing event
r = eng.analyze_event({"kind": "arp_spoof", "ip": "192.168.1.1", "old_mac": "aa:bb:cc:00:11:22",
                       "new_mac": "de:ad:be:ef:00:01"})
ok = r["attack_type"] == "ARP Spoofing"
(PASS if ok else FAIL).append("ARP spoofing")
print(f"\n=== ARP spoofing -> {'PASS' if ok else 'FAIL'}: severity {r['severity']} {r['severity_band']} | {r['why']}")

# normal browsing must stay quiet (false-alarm check)
quiet = [eng.analyze_flow(flow("192.168.1.20", f"93.184.216.{i % 5 + 1}", 443, 900 + i * 3, fwd=14, bwd=18,
                               fb=1800, bb=24000, dur=2.5)) for i in range(12)]
alerts = [q for q in quiet if q["predicted_label"] == "attack"]
print(f"\n=== Normal HTTPS browsing: {len(alerts)} alert(s) from {len(quiet)} flows")
(PASS if len(alerts) <= 1 else FAIL).append("normal traffic stays quiet")

print("\nSample full record (JSON):")
sample = next(e for e in [eng.analyze_flow(flow("192.168.1.99", VIC, 1000 + i, 1000 + i * 0.1, fwd=1, bwd=1, fb=0,
                                                bb=0, dur=0.001, syn=1, ack=0)) for i in range(30)]
              if e["predicted_label"] == "attack")
sample.pop("features", None)
print(json.dumps(sample, indent=2, default=str)[:1500])

rl_used = eng.rl_states > 0
print(f"\nRL policy states loaded: {eng.rl_states} -> {'OK' if rl_used else 'WARNING: actions come from fallback rule'}")
(PASS if rl_used else FAIL).append("RL policy loaded")

print(f"\nRESULT: {len(PASS)} passed, {len(FAIL)} failed  {FAIL if FAIL else ''}")
sys.exit(1 if FAIL else 0)
