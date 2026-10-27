"""
Live capture agent: real packets -> bidirectional flows -> detection service.

  sudo -E env "PATH=$PATH" python api/live_agent.py            (macOS/Linux, needs root to sniff)

Env: LIVE_API_URL (default http://localhost:8000), IFACE (default: system default),
     IDLE_TIMEOUT (default 5 s), VERBOSE=1 to print every flow.

Flow semantics follow CIC-IDS-2017 (the training data): one flow = both directions of a
connection; 'forward' = the side that sent the first packet; sizes are payload bytes;
a flow ends on FIN/RST (after a short grace), after IDLE_TIMEOUT of silence, or at 120 s.
"""
import ipaddress
import os
import queue
import sys
import threading
import time

API = os.environ.get("LIVE_API_URL", "http://localhost:8000")
IDLE_TIMEOUT = float(os.environ.get("IDLE_TIMEOUT", "5"))
MAX_FLOW_SECONDS = 120.0
CLOSE_GRACE = 1.0
VERBOSE = os.environ.get("VERBOSE") == "1"

FIN, SYN, RST, PSH, ACK = 0x01, 0x02, 0x04, 0x08, 0x10


class FlowTable:
    """Pure data structure: feed it packets, ask it for finished flows. Testable without a network."""

    def __init__(self):
        self.flows = {}
        self.closed = {}
        self.lock = threading.Lock()

    @staticmethod
    def parse(pkt):
        """Return (ip_version, src, dst, proto, sport, dport, payload_len, tcp_flags) or None."""
        from scapy.layers.inet import IP, TCP, UDP
        from scapy.layers.inet6 import IPv6
        if pkt.haslayer(IP):
            ip, ver = pkt[IP], "v4"
        elif pkt.haslayer(IPv6):
            ip, ver = pkt[IPv6], "v6"
        else:
            return None
        src, dst = ip.src, ip.dst
        if pkt.haslayer(TCP):
            l4 = pkt[TCP]
            return ver, src, dst, "TCP", int(l4.sport), int(l4.dport), len(l4.payload), int(l4.flags)
        if pkt.haslayer(UDP):
            l4 = pkt[UDP]
            return ver, src, dst, "UDP", int(l4.sport), int(l4.dport), len(l4.payload), 0
        return ver, src, dst, "OTHER", 0, 0, len(ip.payload), 0

    def add(self, parsed, now):
        ver, src, dst, proto, sport, dport, plen, flags = parsed
        try:
            a_src, a_dst = ipaddress.ip_address(src), ipaddress.ip_address(dst)
        except ValueError:
            return
        if a_src.is_loopback or a_dst.is_loopback or a_dst.is_multicast or a_dst == ipaddress.ip_address("255.255.255.255") \
                or a_src.is_unspecified:
            return
        a, b = (src, sport), (dst, dport)
        key = (proto, min(a, b), max(a, b))
        with self.lock:
            if key in self.closed and now < self.closed[key]:
                return
            f = self.flows.get(key)
            if f is None:
                f = {"src_ip": src, "dst_ip": dst, "src_port": sport, "dst_port": dport, "protocol": proto,
                     "ip_version": ver, "start_ts": now, "last_ts": now, "fwd_packets": 0, "bwd_packets": 0,
                     "fwd_bytes": 0, "bwd_bytes": 0, "syn_count": 0, "fin_count": 0, "rst_count": 0,
                     "psh_count": 0, "ack_count": 0, "close_at": None}
                self.flows[key] = f
            if (src, sport) == (f["src_ip"], f["src_port"]):
                f["fwd_packets"] += 1
                f["fwd_bytes"] += plen
            else:
                f["bwd_packets"] += 1
                f["bwd_bytes"] += plen
            f["last_ts"] = now
            if flags & SYN: f["syn_count"] += 1
            if flags & FIN: f["fin_count"] += 1
            if flags & RST: f["rst_count"] += 1
            if flags & PSH: f["psh_count"] += 1
            if flags & ACK: f["ack_count"] += 1
            if flags & (FIN | RST) and f["close_at"] is None:
                f["close_at"] = now + CLOSE_GRACE

    def pop_ready(self, now):
        done = []
        with self.lock:
            for key, f in list(self.flows.items()):
                if ((f["close_at"] is not None and now >= f["close_at"])
                        or now - f["last_ts"] >= IDLE_TIMEOUT
                        or now - f["start_ts"] >= MAX_FLOW_SECONDS):
                    del self.flows[key]
                    self.closed[key] = now + 2.0
                    out = {k: v for k, v in f.items() if k not in ("last_ts", "close_at")}
                    out["duration_sec"] = max(f["last_ts"] - f["start_ts"], 0.0)
                    done.append(out)
            for key in [k for k, t in self.closed.items() if now > t]:
                del self.closed[key]
        return done


class ArpMonitor:
    def __init__(self):
        self.table = {}

    def observe(self, ip, mac):
        if ip in ("0.0.0.0", "") or mac in ("00:00:00:00:00:00", "ff:ff:ff:ff:ff:ff"):
            return None
        old = self.table.get(ip)
        self.table[ip] = mac
        if old is not None and old != mac:
            return {"kind": "arp_spoof", "ip": ip, "old_mac": old, "new_mac": mac}
        return None


def poster(q, stats):
    import requests
    sess = requests.Session()
    while True:
        path, payload = q.get()
        try:
            r = sess.post(API + path, json=payload, timeout=10)
            r.raise_for_status()
            rec = r.json()
            stats["sent"] += 1
            if rec.get("predicted_label") == "attack":
                stats["alerts"] += 1
                print(f"[ALERT] sev={rec['severity']}/10 {rec['severity_band']:8s} {rec['attack_type']:32s} "
                      f"{rec['src_ip']} -> {rec['dst_ip']}:{rec['dst_port']}  action={rec['action_taken']}  "
                      f"layers={','.join(rec['detection_layers'])}", flush=True)
            elif VERBOSE:
                print(f"[flow ] benign {rec['src_ip']} -> {rec['dst_ip']}:{rec['dst_port']} {rec['latency_ms']}ms", flush=True)
        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] % 20 == 1:
                print(f"[agent] cannot reach {API}: {type(e).__name__} (is the service running?)", flush=True)
        finally:
            q.task_done()


def main():
    try:
        from scapy.all import sniff
        from scapy.layers.l2 import ARP
    except ImportError:
        sys.exit("scapy is required: pip install scapy")
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        sys.exit("Packet capture needs root. Run with: sudo -E env \"PATH=$PATH\" python api/live_agent.py")

    table, arp, q = FlowTable(), ArpMonitor(), queue.Queue(maxsize=5000)
    stats = {"sent": 0, "alerts": 0, "errors": 0, "packets": 0}
    threading.Thread(target=poster, args=(q, stats), daemon=True).start()

    def flusher():
        while True:
            time.sleep(1)
            for f in table.pop_ready(time.time()):
                try:
                    q.put_nowait(("/ingest_live", f))
                except queue.Full:
                    pass

    threading.Thread(target=flusher, daemon=True).start()

    def status():
        while True:
            time.sleep(30)
            print(f"[agent] packets={stats['packets']} flows_sent={stats['sent']} alerts={stats['alerts']} "
                  f"open_flows={len(table.flows)} errors={stats['errors']}", flush=True)

    threading.Thread(target=status, daemon=True).start()

    def handle(pkt):
        stats["packets"] += 1
        if pkt.haslayer(ARP):
            a = pkt[ARP]
            if a.op == 2:  # is-at (reply)
                ev = arp.observe(a.psrc, a.hwsrc)
                if ev:
                    try:
                        q.put_nowait(("/ingest_event", ev))
                    except queue.Full:
                        pass
            return
        parsed = FlowTable.parse(pkt)
        if parsed:
            table.add(parsed, time.time())

    iface = os.environ.get("IFACE") or None
    print(f"Capturing on {iface or 'default interface'} -> {API}  (Ctrl+C to stop)", flush=True)
    sniff(prn=handle, store=False, iface=iface, filter="ip or ip6 or arp")


if __name__ == "__main__":
    main()
