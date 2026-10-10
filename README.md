# Wireless Triage Pipeline

Automated root-cause triage for BPSK / QPSK / 16-QAM wireless test cases.  
Upload real IQ data **or** simulate a signal — get a diagnosis instantly.

## Stack

Python · scikit-learn · NumPy · Pandas · Matplotlib · FastAPI · Streamlit · Docker · GitHub Actions

---

## Project Structure

```
wireless-triage/
├── signal_generator.py        # Modulation + impairment simulation + feature extraction
├── test_cases.py              # Pass/fail test cases → test log CSV
├── ml_impairment_detector.py  # RandomForest: RF features → PASS/FAIL
├── ml_failure_categorizer.py  # TF-IDF + Logistic Regression: log text → failure category
├── triage_engine.py           # Root-cause engine combining both models
├── iq_processor.py            # Real IQ file parser + blind feature extraction
├── api.py                     # FastAPI REST service (includes IQ upload endpoint)
├── ui.py                      # Streamlit UI (upload tab + simulator tab)
├── visualize.py               # Exploratory plots
├── train_all.py               # One-shot bootstrap script
├── supervisord.conf           # Runs FastAPI + Streamlit together in Docker
├── requirements.txt
├── Dockerfile
├── .github/workflows/ci.yml   # GitHub Actions CI/CD
└── tests/
    ├── test_signal_generator.py
    ├── test_iq_processor.py
    └── test_api.py
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Bootstrap (generate data + train models)

```bash
python train_all.py
```

### 3. Run the Streamlit UI

```bash
streamlit run ui.py
```

Opens at http://localhost:8501 — two tabs:
- **Upload Real IQ Data** — upload a `.csv/.npy/.bin/.txt` file from your SDR or analyser
- **Simulate Signal** — tune SNR, frequency offset, interference with sliders

### 4. Run the REST API

```bash
uvicorn api:app --reload
```

Interactive docs: http://localhost:8000/docs

---

## Real IQ File Upload

### Supported formats

| Format | Description |
|--------|-------------|
| `.csv` | Two columns: `I` and `Q` (or `real`/`imag`) |
| `.npy` | NumPy complex array or shape `(N, 2)` float |
| `.bin` | Raw interleaved float32: I₀ Q₀ I₁ Q₁ … |
| `.txt` | Whitespace-separated I Q pairs, one per line |

### Get a sample file

```bash
# Download a sample QPSK IQ CSV at 15 dB SNR
curl "http://localhost:8000/triage/sample-iq?modulation=QPSK&snr_db=15" -o sample.csv
```

Or click the **Download sample file** button in the UI.

### Upload via API

```bash
curl -X POST http://localhost:8000/triage/iq-upload \
  -F "file=@my_capture.csv" \
  -F "modulation=QPSK"
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness + model readiness |
| POST | `/triage/case` | Full triage on simulated test case |
| POST | `/triage/iq-upload` | **Upload real IQ file → diagnosis** |
| GET | `/triage/sample-iq` | Download sample IQ CSV |
| POST | `/triage/batch` | Batch triage over `data/test_log.csv` |
| GET | `/triage/summary` | Pre-computed summary report |
| GET | `/triage/report` | Full per-case JSON report |
| POST | `/predict/impairment` | Direct RF impairment detection |
| POST | `/predict/category` | Log message failure classification |

---

## Docker (runs both UI + API)

```bash
# Build
docker build -t wireless-triage .

# Run
docker run -p 8000:8000 -p 8501:8501 wireless-triage
```

- UI:  http://localhost:8501
- API: http://localhost:8000/docs

---

## Models

| Model | Algorithm | Input | Target |
|-------|-----------|-------|--------|
| Impairment Detector | Random Forest | Signal features (EVM, BER, phase std, …) | PASS / FAIL |
| Failure Categorizer | TF-IDF + Logistic Regression | Test log message text | HIGH_EVM / HIGH_BER / PHASE_NOISE / MULTIPLE / PASS |

---

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`):
1. Lint with flake8
2. Run pytest suite
3. On merge to `main`: build & push Docker image to GHCR
