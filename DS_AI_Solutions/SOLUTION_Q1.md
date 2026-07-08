# Question 1: Predicting Enterprise Server Failure Across Global Data Centers

## My Reading of the Problem

Reading through this, scale jumped out at me right away - ten clients, up to 20 data centers each, 5,000+ servers per client. That rules out anything hand-tuned from the start. But the part I kept coming back to was the 40% downtime reduction target, because that's easy to underestimate. Detecting failures isn't enough on its own; you need enough lead time to actually do something before the server goes down.

The other thing that kept nagging at me was the multi-client setup. Some clients run high-memory database servers where 85% memory utilization is completely normal. Others have web servers where the same number means trouble. A global threshold won't work - the model has to learn per-environment baselines on its own.

## My Approach

I went with a two-model ensemble with an adaptive layer sitting on top of it.

For tabular features, XGBoost does the heavy lifting - it handles structured server telemetry well, trains fast, and its feature importances give the ops team something concrete when they need to explain an alert to a client. For temporal patterns (gradual degradation that plays out over hours), I added a bidirectional LSTM that looks at 24-hour sequences of rolling metrics. The two models score independently and their outputs are blended - 60% XGBoost, 40% LSTM. XGBoost gets the higher weight because it's more reliable on short or sparse histories; the LSTM needs a decent sequence length to add real signal.

Here's the overall flow:

```text
Raw telemetry (per server, per hour)
          |
          v
Missing value handling (forward-fill + KNN)
          |
          v
Feature engineering (rolling stats, anomaly scores, rate of change)
          |
          v
XGBoost + LSTM ensemble scoring
          |
          v
Risk tier assignment + maintenance scheduling
```

## Handling Missing Logs

The code uses a two-stage approach. For time-ordered telemetry (CPU, memory, disk I/O, latency), it forward-fills then backward-fills per server. This preserves the trend - if memory was at 78% and the next reading is missing, forward-filling from 78% is a much better guess than replacing with a global median. After that, anything still missing gets filled by KNN imputation (k=5, distance-weighted) from the nearest similar servers.

There's also a hard validation step at ingestion: if the missing rate across any column exceeds 25%, the pipeline raises an error rather than silently training on garbage data.

## Feature Engineering

The feature step creates 80+ features per server:

- Rolling mean, std, max, and range over 1h, 6h, and 24h windows for each core metric
- Velocity (first difference) and acceleration (second difference) per metric - to catch gradual degradation even when absolute values look fine
- Log-derived indicators: error ratio, crash density over 24h, restart spike flags
- Days since last maintenance and normalized historical failure rate
- Isolation Forest anomaly score as an unsupervised signal on top of the supervised features

The anomaly score is worth calling out - it catches failure patterns the labeled training data hasn't seen before. Isolation Forest flags outlier behavior regardless of whether there's a historical label for that particular failure mode.

## Adapting to New Clients

When a new client or infrastructure setup comes in, the system starts from the global model weights. The adaptive layer monitors accuracy using ADWIN (Adaptive Windowing) drift detection. When ADWIN detects a significant shift - meaning the global model is performing worse than expected on this client's data - it buffers recent labeled observations and triggers a fine-tune of the XGBoost model on that buffer. Usually takes 24-48 hours of data before the client-specific adaptation stabilizes.

Per-client normalizers handle the baseline mismatch problem independently of drift detection: each client's feature distributions are normalized separately before prediction, so a server that normally runs hot doesn't get flagged just for being elevated relative to the global population.

## Getting to 40% Downtime Reduction

The model outputs a 6-hour look-ahead failure probability, bucketed into four risk tiers:

- **CRITICAL** (≥ 70%) → immediate maintenance alert
- **HIGH** (≥ 50%) → schedule within 4 hours
- **MEDIUM** (≥ 35%) → schedule within 24 hours
- **LOW** → routine scheduling, no priority bump

The proactive scheduler enforces a concurrency cap (default 10 simultaneous maintenance windows) so the ops team doesn't get flooded all at once. The 6-hour lead time is what makes the difference - that's roughly the window where you can actually intervene before a failure cascades into extended downtime. I'd expect something in the 40%+ range in practice, though that depends a lot on how consistently the ops team follows up on alerts. A model that gets ignored half the time won't hit any reduction target regardless of how accurate the predictions are.

The classification threshold is tuned to 0.35 rather than the usual 0.5, biasing toward recall - missing a real failure is much more expensive than a false alarm.

## Keeping Prediction Under 5 Seconds

XGBoost inference on pre-computed features is fast - batch prediction on a few hundred servers typically finishes in well under a second. The LSTM adds some overhead, but the total stays comfortably below the 5-second mark. If this were going to production at full scale, the natural next step would be pre-computing and caching feature vectors in a feature store, so the prediction endpoint only has to run the model itself - that's where you'd start seeing meaningful gains when you're serving thousands of servers concurrently.

## Running the Code

```bash
python -m pip install -r ../requirements_q1.txt
python Q1_server_failure_prediction.py
```

The demo runs on 100 servers over 48 hours of synthetic telemetry, trains the full ensemble (LSTM optional - degrades gracefully if TensorFlow isn't installed), generates per-server predictions, and outputs a prioritized maintenance schedule.
