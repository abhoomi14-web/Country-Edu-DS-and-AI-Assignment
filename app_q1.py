import sys
from pathlib import Path

# Make DS_AI_Solutions importable as a module
_root = Path(__file__).parent
sys.path.insert(0, str(_root / "DS_AI_Solutions"))

import joblib
import threading
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from Q1_server_failure_prediction import (
    generate_synthetic_data,
    EnsembleServerFailurePredictor,
    ProactiveMaintenanceScheduler,
)

app = FastAPI(title="Server Failure Prediction Dashboard", version="1.0.0")

_PKL = _root / "trained_predictor_web.pkl"

_predictor: EnsembleServerFailurePredictor | None = None
_lock = threading.Lock()


def get_predictor() -> EnsembleServerFailurePredictor:
    global _predictor
    with _lock:
        if _predictor is None:
            if _PKL.exists():
                try:
                    _predictor = joblib.load(str(_PKL))
                except Exception:
                    _predictor = None
            if _predictor is None:
                print("[Q1] Training predictor (first run — takes ~15s)…")
                df, labels = generate_synthetic_data(n_servers=100, n_hours=48)
                _predictor = EnsembleServerFailurePredictor()
                _predictor.fit(df, labels)
                joblib.dump(_predictor, str(_PKL))
                print("[Q1] Model trained and cached.")
    return _predictor


@app.on_event("startup")
def preload():
    get_predictor()


app.mount(
    "/dashboard",
    StaticFiles(directory=str(_root / "static_q1"), html=True),
    name="dashboard",
)


@app.get("/")
def root():
    return RedirectResponse(url="/dashboard/")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run-demo")
def run_demo():
    predictor = get_predictor()

    # Generate a fresh snapshot to run through the loaded model
    df, _ = generate_synthetic_data(n_servers=100, n_hours=48)
    snapshot = df.groupby("server_id").tail(1).copy()

    predictions = predictor.predict(snapshot)
    scheduler = ProactiveMaintenanceScheduler()
    schedule_df = scheduler.build_schedule(predictions)

    rows = []
    for _, row in schedule_df.iterrows():
        rows.append({
            "server_id": row["server_id"],
            "failure_probability": round(float(row["failure_probability"]), 4),
            "risk_tier": row["risk_tier"],
            "scheduled_maintenance": str(row["scheduled_maintenance"]),
            "alert_immediately": bool(row["alert_immediately"]),
            "estimated_downtime_prevented_h": round(float(row["estimated_downtime_prevented_h"]), 2),
            "maintenance_slot": int(row["maintenance_slot"]),
        })

    tier_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    total_downtime_prevented = 0.0
    for r in rows:
        tier_counts[r["risk_tier"]] = tier_counts.get(r["risk_tier"], 0) + 1
        total_downtime_prevented += r["estimated_downtime_prevented_h"]

    return {
        "servers_evaluated": len(rows),
        "tier_counts": tier_counts,
        "total_downtime_prevented_h": round(total_downtime_prevented, 1),
        "schedule": rows,
    }
