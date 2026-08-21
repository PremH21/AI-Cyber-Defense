"""
Live Packet-Capture Agent — captures real network traffic (IPv4 + IPv6)
from this machine, extracts flow-level features, and reports summarized
flows periodically regardless of traffic timing.
"""
import time
import threading
import requests
from collections import defaultdict
from scapy.all import sniff, IP, IPv6, TCP, UDP

API_URL = "http://localhost:8000/ingest_live"
FLOW_TIMEOUT = 3
FLUSH_INTERVAL = 2

flows = defaultdict(lambda: {
    "packets": [], "start_time": None, "last_time": None,
})
lock = threading.Lock()


def flow_key(pkt):
    if pkt.haslayer(IP):
        ip_layer = pkt[IP]
        version = "v4"
    elif pkt.haslayer(IPv6):
        ip_layer = pkt[IPv6]
        version = "v6"
    else:
        return None

    proto = "TCP" if pkt.haslayer(TCP) else "UDP" if pkt.haslayer(UDP) else "OTHER"
    sport = pkt[TCP].sport if pkt.haslayer(TCP) else (pkt[UDP].sport if pkt.haslayer(UDP) else 0)
    dport = pkt[TCP].dport if pkt.haslayer(TCP) else (pkt[UDP].dport if pkt.haslayer(UDP) else 0)
    return (ip_layer.src, ip_layer.dst, sport, dport, proto, version)


def process_packet(pkt):
    key = flow_key(pkt)
    if key is None:
        return
    now = time.time()
    with lock:
        f = flows[key]
        if f["start_time"] is None:
            f["start_time"] = now
        f["last_time"] = now
        f["packets"].append(len(pkt))


def report_flow(key, flow):
    src, dst, sport, dport, proto, version = key
    duration = max(flow["last_time"] - flow["start_time"], 0.0001)
    n_packets = len(flow["packets"])
    total_bytes = sum(flow["packets"])

    summary = {
        "src_ip": src, "dst_ip": dst, "src_port": sport, "dst_port": dport,
        "protocol": proto, "ip_version": version, "duration_sec": round(duration, 4),
        "packet_count": n_packets, "total_bytes": total_bytes,
        "avg_packet_size": round(total_bytes / n_packets, 2) if n_packets else 0,
        "packets_per_sec": round(n_packets / duration, 2),
    }

    print(f"[FLOW] {src}:{sport} -> {dst}:{dport} ({proto}/{version})  "
          f"{n_packets} pkts, {total_bytes} bytes, {duration:.2f}s")

    try:
        resp = requests.post(API_URL, json=summary, timeout=3)
        if resp.status_code == 200:
            result = resp.json()
            print(f"       -> verdict={result.get('predicted_label')}  "
                  f"action={result.get('action_taken')}")
    except requests.exceptions.ConnectionError:
        pass  # backend not running yet — expected during capture-only testing
    except Exception as e:
        print(f"       -> WARNING: {e}")


def flush_loop():
    while True:
        time.sleep(FLUSH_INTERVAL)
        now = time.time()
        with lock:
            stale_keys = [k for k, v in flows.items()
                          if v["last_time"] is not None and now - v["last_time"] > FLOW_TIMEOUT]
            to_report = [(k, flows.pop(k)) for k in stale_keys]
        for k, flow in to_report:
            report_flow(k, flow)


if __name__ == "__main__":
    print("Live Packet-Capture Agent starting (IPv4 + IPv6)...")
    print(f"Flushing completed flows every {FLUSH_INTERVAL}s (timeout={FLOW_TIMEOUT}s of inactivity).")
    print("Press Ctrl+C to stop.\n")

    flusher = threading.Thread(target=flush_loop, daemon=True)
    flusher.start()

    try:
        sniff(prn=process_packet, store=False)
    except KeyboardInterrupt:
        print("\nAgent stopped.")
    except PermissionError:
        print("ERROR: run with sudo (macOS/Linux) or as Administrator (Windows, with Npcap installed).")
