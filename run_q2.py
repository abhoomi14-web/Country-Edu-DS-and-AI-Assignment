import socket
import threading
import time
import webbrowser
import urllib.request
import uvicorn

def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def open_when_ready(url):
    health = url + "/health"
    for _ in range(150):  # wait up to 30s, check every 0.2s
        try:
            urllib.request.urlopen(health, timeout=1)
            webbrowser.open(url)
            return
        except Exception:
            time.sleep(0.2)

if __name__ == "__main__":
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"\n  IT Ticket Auto-Resolution Dashboard")
    print(f"  Opening: {url}")
    print(f"  API docs: {url}/docs\n")
    threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()
    uvicorn.run("app_q2:app", host="127.0.0.1", port=port, reload=False)
