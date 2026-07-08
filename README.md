# DS & AI Internship — Final Submission

Two solutions for the DS & AI problem set, covering server failure prediction and IT ticket auto-resolution.

## Structure

```
Bhoomi FInal Submission/
├── requirements_q1.txt               — dependencies for Q1
├── requirements_q2.txt               — dependencies for Q2
└── DS_AI_Solutions/
    ├── Q1_server_failure_prediction.py   — server failure prediction pipeline
    ├── Q2_ticket_auto_resolution.py      — ticket classification and resolution
    ├── SOLUTION_Q1.md                    — writeup and design rationale for Q1
    └── SOLUTION_Q2.md                    — writeup and design rationale for Q2
```

## Running Q1

Requires Python 3.10+.

```bash
python -m pip install -r requirements_q1.txt
python DS_AI_Solutions/Q1_server_failure_prediction.py
```

The demo generates synthetic telemetry for 100 servers over 48 hours, trains the ensemble, runs predictions on the final snapshot per server, and outputs a maintenance schedule.

## Running Q2

```bash
python -m pip install -r requirements_q2.txt
python DS_AI_Solutions/Q2_ticket_auto_resolution.py
```

Demo runs in TF-IDF mode by default for speed. Swap `use_bert=False` to `use_bert=True` in the `__main__` block for the full DistilBERT run.

## Notes

- TensorFlow (Q1 LSTM) and PyTorch (Q2 DistilBERT) are optional — both scripts fall back gracefully if they're not installed.
- Tesseract OCR for Q2 screenshot tickets: install Tesseract separately and set the binary path in `ScreenshotOCRExtractor`.
- In production, replace the in-memory `ResponseCache` in Q2 with Redis and load the FAISS index from disk.
