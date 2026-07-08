import sys
from pathlib import Path

_root = Path(__file__).parent
sys.path.insert(0, str(_root / "DS_AI_Solutions"))

import joblib
import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from Q2_ticket_auto_resolution import (
    TicketAutoResolutionSystem,
    generate_synthetic_tickets,
    generate_synthetic_kb,
)

app = FastAPI(title="IT Ticket Auto-Resolution Dashboard", version="1.0.0")

_PKL = _root / "trained_resolution_system_web.pkl"

_system: TicketAutoResolutionSystem | None = None
_lock = threading.Lock()


def get_system() -> TicketAutoResolutionSystem:
    global _system
    with _lock:
        if _system is None:
            if _PKL.exists():
                try:
                    _system = joblib.load(str(_PKL))
                except Exception:
                    _system = None
            if _system is None:
                print("[Q2] Training resolution system (first run — takes ~5s)…")
                tickets, _ = generate_synthetic_tickets(n=500)
                kb_entries = generate_synthetic_kb()
                _system = TicketAutoResolutionSystem(use_bert=False)
                _system.fit(tickets[:400], kb_entries)
                joblib.dump(_system, str(_PKL))
                print("[Q2] System trained and cached.")
    return _system


# Pre-load model at startup so the first request isn't slow
@app.on_event("startup")
def preload():
    get_system()


app.mount(
    "/dashboard",
    StaticFiles(directory=str(_root / "static_q2"), html=True),
    name="dashboard",
)


@app.get("/")
def root():
    return RedirectResponse(url="/dashboard/")


@app.get("/health")
def health():
    status = "ready" if _system is not None else "loading"
    return {"status": status}


class TicketRequest(BaseModel):
    text: str
    ticket_id: str = "WEB-001"


@app.post("/resolve")
def resolve_ticket(body: TicketRequest):
    system = get_system()
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Ticket text cannot be empty.")

    result = system.resolve({"id": body.ticket_id, "description": body.text})
    return result
