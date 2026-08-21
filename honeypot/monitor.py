"""
Honeypot Grid — watches decoy files for modification, deletion, or renaming,
which is what ransomware actually does to files (encrypt = modify,
mass-rename with new extension). File *reads* are intentionally not
monitored: macOS's native file-event API doesn't reliably report simple
opens, so relying on it would give false confidence.
"""
import time
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

API_URL = "http://localhost:8000/honeypot_alert"
WATCH_DIR = "honeypot/decoys"


class HoneypotHandler(FileSystemEventHandler):
    def _alert(self, path, event_type):
        filename = path.split("/")[-1]
        print(f"\n🚨 HONEYPOT TRIGGERED: {filename} was {event_type}")
        try:
            resp = requests.post(API_URL, json={"decoy_file": filename, "event_type": event_type}, timeout=5)
            result = resp.json()
            print(f"   -> Action taken: {result.get('action_taken')}  (logged to MongoDB)")
        except requests.exceptions.ConnectionError:
            print("   -> ERROR: API not reachable. Is uvicorn running?")

    def on_modified(self, event):
        if not event.is_directory:
            self._alert(event.src_path, "modified")

    def on_deleted(self, event):
        if not event.is_directory:
            self._alert(event.src_path, "deleted")

    def on_moved(self, event):
        if not event.is_directory:
            self._alert(event.dest_path, "renamed")


if __name__ == "__main__":
    print(f"Honeypot Grid active — watching {WATCH_DIR}/ for modification, deletion, or renaming.")
    print("(Ransomware-style behavior: encrypting/renaming/deleting decoy files.)")
    print("Ctrl+C to stop.\n")

    observer = Observer()
    observer.schedule(HoneypotHandler(), WATCH_DIR, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
