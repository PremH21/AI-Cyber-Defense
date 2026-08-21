import subprocess
import time
from collections import deque
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

WATCH_DIR = "response-engine/protected_files"
RAPID_CHANGE_THRESHOLD = 5   # N modifications
RAPID_CHANGE_WINDOW = 10     # within N seconds -> looks like ransomware

def take_snapshot(reason="scheduled"):
    print(f"[{datetime.now().isoformat()}] Taking APFS local snapshot ({reason})...")
    result = subprocess.run(
        ["tmutil", "localsnapshot"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  Snapshot created: {result.stdout.strip()}")
    else:
        print(f"  Snapshot FAILED: {result.stderr.strip()}")
    return result.returncode == 0

def list_snapshots():
    result = subprocess.run(["tmutil", "listlocalsnapshots", "/"], capture_output=True, text=True)
    print("Current local snapshots:")
    print(result.stdout)

class RansomwareLikeHandler(FileSystemEventHandler):
    def __init__(self):
        self.recent_events = deque()

    def on_modified(self, event):
        self._log_event(event)

    def on_created(self, event):
        self._log_event(event)

    def _log_event(self, event):
        now = time.time()
        self.recent_events.append(now)
        while self.recent_events and now - self.recent_events[0] > RAPID_CHANGE_WINDOW:
            self.recent_events.popleft()

        print(f"[{datetime.now().isoformat()}] Change detected: {event.src_path}")

        if len(self.recent_events) >= RAPID_CHANGE_THRESHOLD:
            print(f"  ALERT: {len(self.recent_events)} changes in {RAPID_CHANGE_WINDOW}s "
                  f"- ransomware-like pattern. Triggering emergency snapshot.")
            take_snapshot(reason="emergency - suspected ransomware activity")
            self.recent_events.clear()

def main():
    print("=== Shadow Copy Guardian (macOS APFS snapshots) ===")
    take_snapshot(reason="startup baseline")
    list_snapshots()

    event_handler = RansomwareLikeHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIR, recursive=True)
    observer.start()
    print(f"\nWatching {WATCH_DIR} for suspicious rapid file changes...")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopped. Final snapshot list:")
        list_snapshots()
    observer.join()

if __name__ == "__main__":
    main()
