# Wireless Triage Pipeline

Automated root-cause triage for BPSK / QPSK / 16-QAM wireless test cases.

## Stack

Python · scikit-learn · NumPy · Pandas · Matplotlib · FastAPI · Docker · GitHub Actions

---

## Project Structure

```
wireless-triage/
├── signal_generator.py        # Modulation + impairment simulation + feature extraction
├── test_cases.py              # Pass/fail test cases → test log CSV
├── ml_impairment_detector.py  # RandomForest: RF features → PASS/FAIL
├── ml_failure_categorizer.py  # TF-IDF + Logistic Regression: log text → failure category
├── triage_engine.py           # Root-cause engine combining both models
├── api.py                     # FastAPI REST service
├── visualize.py               # Exploratory plots
├── train_all.py               # One-shot bootstrap script
├── requirements.txt
├── Dockerfile
├── .github/workflows/ci.yml   # GitHub Actions CI/CD
└── tests/
    ├── test_signal_generator.py
    └── test_api.py
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Bootstrap (generate data + train models + initial triage)

```bash
python train_all.py
```

### 3. Run the API

```bash
uvicorn api:app --reload
```

Interactive docs: http://localhost:8000/docs

### 4. Generate visualizations

```bash
python visualize.py
```

---

## Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness + model readiness check |
| POST | `/triage/case` | Full triage on one test case |
| POST | `/triage/batch` | Batch triage over `data/test_log.csv` |
| GET | `/triage/summary` | Pre-computed summary report |
| GET | `/triage/report` | Full per-case JSON report |
| POST | `/predict/impairment` | Direct RF impairment detection |
| POST | `/predict/category` | Log message failure classification |

### Example: triage a single case

```bash
curl -X POST http://localhost:8000/triage/case \
  -H "Content-Type: application/json" \
  -d '{
    "modulation": "QPSK",
    "snr_db": 10.0,
    "freq_offset": 0.03,
    "interference_power": 0.2,
    "case_id": 1
  }'
```

---

## Docker

```bash
# Build (trains models during build)
docker build -t wireless-triage .

# Run
docker run -p 8000:8000 wireless-triage
```

---

## Models

| Model | Algorithm | Input | Target |
|-------|-----------|-------|--------|
| Impairment Detector | Random Forest | Signal features (EVM, BER, phase std, …) | PASS / FAIL |
| Failure Categorizer | TF-IDF + Logistic Regression | Test log message text | HIGH_EVM / HIGH_BER / PHASE_NOISE / MULTIPLE / PASS |

---

## Impairments Simulated

| Impairment | Parameter |
|------------|-----------|
| AWGN | SNR: 5–30 dB |
| Frequency offset | 0–0.05 (normalised) |
| Sinusoidal interference | Power: 0–0.3 |

---

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`):
1. Lint with flake8
2. Run pytest suite
3. On merge to `main`: build & push Docker image to GHCR
