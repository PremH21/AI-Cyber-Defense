"""
Shadow Copy Guardian (Bonus #6)

Monitors a target directory and takes rolling snapshots on an interval.
If ransomware-like behavior is detected (mass file deletion or rapid
modification within a short window — the "shadow copy deletion" step
ransomware typically performs first), it automatically restores files
from the most recent good snapshot.

This is a local filesystem simulation of the VSS (Volume Shadow Copy
Service) rollback described in the slide deck — same concept, without
requiring actual Windows VSS APIs, so it runs cross-platform for the demo.
"""

import os
import shutil
import time
import hashlib
import json
from datetime import datetime

WATCHED_DIR = "response-engine/demo_data/watched_folder"
SNAPSHOT_DIR = "response-engine/demo_data/snapshots"
LOG_PATH = "response-engine/demo_data/guardian_log.json"

SNAPSHOT_INTERVAL_SECONDS = 5
MAX_SNAPSHOTS_TO_KEEP = 6
# If more than this fraction of files vanish/change between two snapshot
# checks, treat it as a ransomware-style mass-encryption/deletion event.
MASS_CHANGE_THRESHOLD = 0.5


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def snapshot_state(directory):
    state = {}
    if not os.path.exists(directory):
        return state
    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        if os.path.isfile(fpath):
            state[fname] = file_hash(fpath)
    return state


def take_snapshot():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_path = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}")
    if os.path.exists(WATCHED_DIR) and os.listdir(WATCHED_DIR):
        shutil.copytree(WATCHED_DIR, snap_path)
        return snap_path
    return None


def prune_old_snapshots():
    if not os.path.exists(SNAPSHOT_DIR):
        return
    snaps = sorted(os.listdir(SNAPSHOT_DIR))
    while len(snaps) > MAX_SNAPSHOTS_TO_KEEP:
        oldest = snaps.pop(0)
        shutil.rmtree(os.path.join(SNAPSHOT_DIR, oldest))


def most_recent_good_snapshot():
    if not os.path.exists(SNAPSHOT_DIR):
        return None
    snaps = sorted(os.listdir(SNAPSHOT_DIR))
    return os.path.join(SNAPSHOT_DIR, snaps[-1]) if snaps else None


def restore_from_snapshot(snap_path):
    if os.path.exists(WATCHED_DIR):
        shutil.rmtree(WATCHED_DIR)
    shutil.copytree(snap_path, WATCHED_DIR)


def log_event(event):
    events = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            events = json.load(f)
    events.append({**event, "timestamp": datetime.now().isoformat()})
    with open(LOG_PATH, "w") as f:
        json.dump(events, f, indent=2)


def run_monitor_cycles(n_cycles):
    os.makedirs(WATCHED_DIR, exist_ok=True)
    print(f"Shadow Copy Guardian watching: {WATCHED_DIR}")
    print(f"Snapshot interval: {SNAPSHOT_INTERVAL_SECONDS}s | Mass-change threshold: {int(MASS_CHANGE_THRESHOLD*100)}%\n")

    prev_state = snapshot_state(WATCHED_DIR)
    snap = take_snapshot()
    if snap:
        print(f"[t=0s] Initial snapshot taken: {os.path.basename(snap)} ({len(prev_state)} files)")
        log_event({"action": "snapshot", "path": snap, "file_count": len(prev_state)})

    for cycle in range(1, n_cycles + 1):
        time.sleep(SNAPSHOT_INTERVAL_SECONDS)
        current_state = snapshot_state(WATCHED_DIR)

        if len(prev_state) == 0:
            prev_state = current_state
            continue

        vanished_or_changed = sum(
            1 for fname, old_hash in prev_state.items()
            if fname not in current_state or current_state[fname] != old_hash
        )
        change_fraction = vanished_or_changed / len(prev_state) if prev_state else 0

        print(f"[t={cycle * SNAPSHOT_INTERVAL_SECONDS}s] Checked {len(prev_state)} tracked files — "
              f"{vanished_or_changed} vanished/changed ({change_fraction:.0%})")

        if change_fraction >= MASS_CHANGE_THRESHOLD:
            print(f"  ⚠ RANSOMWARE-STYLE MASS CHANGE DETECTED ({change_fraction:.0%} >= {MASS_CHANGE_THRESHOLD:.0%} threshold)")
            good_snap = most_recent_good_snapshot()
            if good_snap:
                print(f"  -> Auto-restoring from most recent good snapshot: {os.path.basename(good_snap)}")
                restore_from_snapshot(good_snap)
                log_event({
                    "action": "auto_restore",
                    "trigger": "mass_change_detected",
                    "change_fraction": change_fraction,
                    "restored_from": good_snap,
                })
                current_state = snapshot_state(WATCHED_DIR)
                print(f"  -> Restore complete. {len(current_state)} files recovered without paying ransom.")
            else:
                print("  -> No prior snapshot available to restore from.")
        else:
            snap = take_snapshot()
            if snap:
                log_event({"action": "snapshot", "path": snap, "file_count": len(current_state)})
                prune_old_snapshots()

        prev_state = current_state

    print(f"\nDone. Full event log saved to {LOG_PATH}")


if __name__ == "__main__":
    run_monitor_cycles(n_cycles=8)
