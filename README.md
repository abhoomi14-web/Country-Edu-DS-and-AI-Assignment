# DS & AI Internship — Final Submission

Two solutions for the DS & AI problem set: server failure prediction and IT ticket auto-resolution. Both come with an interactive web dashboard that opens automatically in your browser.

## Structure

```
Bhoomi FInal Submission/
├── run_q1.py                             — launch Q1 dashboard
├── run_q2.py                             — launch Q2 dashboard
├── requirements_q1.txt                   — dependencies for Q1
├── requirements_q2.txt                   — dependencies for Q2
├── static_q1/index.html                  — Q1 dashboard UI
├── static_q2/index.html                  — Q2 dashboard UI
└── DS_AI_Solutions/
    ├── Q1_server_failure_prediction.py   — server failure prediction pipeline
    ├── Q2_ticket_auto_resolution.py      — ticket classification and resolution
    ├── SOLUTION_Q1.md                    — writeup and design rationale for Q1
    └── SOLUTION_Q2.md                    — writeup and design rationale for Q2
```

## Q1 — Server Failure Prediction Dashboard

```bash
python -m pip install -r requirements_q1.txt
python run_q1.py
```

The browser opens automatically. Click **Run Prediction** to generate synthetic server telemetry, run the XGBoost + LSTM ensemble, and see the maintenance schedule colour-coded by risk tier (CRITICAL / HIGH / MEDIUM / LOW).

## Q2 — IT Ticket Auto-Resolution Dashboard

```bash
python -m pip install -r requirements_q2.txt
python run_q2.py
```

The browser opens automatically once the model is ready. Type any IT issue into the text box and click **Resolve Ticket** to get a category, confidence score, and ranked solutions from the knowledge base.

## Notes

- Both launchers pick a free port automatically — no port conflicts.
- TensorFlow (Q1 LSTM) and PyTorch (Q2 DistilBERT) are optional. Both scripts fall back gracefully if they are not installed.
- Requires Python 3.10+.
