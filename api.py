"""
api.py
FastAPI application that exposes the wireless triage engine via REST endpoints.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json
import io
import os

from triage_engine import triage_case, triage_batch, build_summary_report, ROOT_CAUSE_MAP, RECOMMENDATIONS
from ml_impairment_detector import predict_impairment
from ml_failure_categorizer import categorize_failure
from iq_processor import (
    parse_iq_file,
    extract_features_blind,
    diagnose_from_blind_features,
    generate_sample_iq_csv,
)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Wireless Triage API",
    description=(
        "Automated root-cause triage for BPSK/QPSK/16-QAM wireless test cases. "
        "Combines RF impairment detection and failure categorization into actionable reports."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TriageCaseRequest(BaseModel):
    modulation: str = Field(..., description="BPSK | QPSK | 16QAM")
    snr_db: float = Field(..., ge=0, le=50, description="Signal-to-noise ratio in dB")
    freq_offset: float = Field(..., ge=0, le=1, description="Normalised frequency offset")
    interference_power: float = Field(..., ge=0, le=10, description="Interference power (linear)")
    case_id: int = Field(0, description="Optional case identifier")

    model_config = {
        "json_schema_extra": {
            "example": {
                "modulation": "QPSK",
                "snr_db": 12.5,
                "freq_offset": 0.02,
                "interference_power": 0.1,
                "case_id": 1,
            }
        }
    }


class ImpairmentRequest(BaseModel):
    modulation: str
    snr_db: float
    freq_offset: float
    interference_power: float
    evm_percent: float
    amp_std: float
    phase_std: float
    kurtosis: float
    mean_amp_error: float


class CategorizerRequest(BaseModel):
    log_message: str = Field(..., description="Raw test log message string")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
def health_check():
    """Check that the API is alive and models are loaded."""
    models_ready = os.path.exists("models/impairment_detector.joblib") and os.path.exists(
        "models/failure_categorizer.joblib"
    )
    return {"status": "ok", "models_ready": models_ready}


@app.post("/triage/case", tags=["Triage"])
def triage_single_case(req: TriageCaseRequest):
    """
    Run a full triage on a single wireless test case.

    Returns: metrics, impairment detection result, failure category,
             root cause analysis, and recommendations.
    """
    valid_mods = {"BPSK", "QPSK", "16QAM"}
    if req.modulation not in valid_mods:
        raise HTTPException(
            status_code=422,
            detail=f"modulation must be one of {sorted(valid_mods)}",
        )
    try:
        report = triage_case(
            modulation=req.modulation,
            snr_db=req.snr_db,
            freq_offset=req.freq_offset,
            interference_power=req.interference_power,
            case_id=req.case_id,
        )
        return report
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/triage/summary", tags=["Triage"])
def get_triage_summary():
    """
    Return the pre-computed summary report from the last batch triage run.
    Requires outputs/summary_report.json to exist.
    """
    path = "outputs/summary_report.json"
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="Summary report not found. Run triage_engine.py first.",
        )
    with open(path) as f:
        return json.load(f)


@app.get("/triage/report", tags=["Triage"])
def get_full_report():
    """
    Return the full per-case triage report JSON from the last batch run.
    Requires outputs/triage_report.json to exist.
    """
    path = "outputs/triage_report.json"
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="Triage report not found. Run triage_engine.py first.",
        )
    with open(path) as f:
        return json.load(f)


@app.post("/triage/batch", tags=["Triage"])
def run_batch_triage():
    """
    Trigger a fresh batch triage over data/test_log.csv.
    Writes outputs/triage_report.json and outputs/summary_report.json.
    """
    if not os.path.exists("data/test_log.csv"):
        raise HTTPException(
            status_code=404,
            detail="data/test_log.csv not found. Run test_cases.py first.",
        )
    reports = triage_batch()
    summary = build_summary_report(reports)
    with open("outputs/summary_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


@app.post("/predict/impairment", tags=["Models"])
def predict_rf_impairment(req: ImpairmentRequest):
    """
    Directly call the RF impairment detector with pre-computed features.
    Returns predicted label (PASS/FAIL) and class probabilities.
    """
    try:
        return predict_impairment(req.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/category", tags=["Models"])
def predict_failure_category(req: CategorizerRequest):
    """
    Classify a test log message string into a failure category.
    Returns the category and per-class probabilities.
    """
    try:
        return categorize_failure(req.log_message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Real IQ data endpoints
# ---------------------------------------------------------------------------

@app.post("/triage/iq-upload", tags=["Real IQ Data"])
async def triage_iq_upload(
    file: UploadFile = File(..., description="IQ data file (.csv, .npy, .bin, .txt)"),
    modulation: str = Form(..., description="BPSK | QPSK | 16QAM"),
):
    """
    Upload a real captured IQ data file and get a full triage report.

    Accepted formats:
    - **CSV**: columns I,Q  or  real,imag  or two unnamed numeric columns
    - **NPY**: numpy complex array or shape (N,2) float array
    - **BIN**: raw interleaved float32 binary (I0,Q0,I1,Q1,...)
    - **TXT**: whitespace-separated I Q pairs, one per line

    Returns the same structured report as /triage/case but driven by real signal data.
    """
    valid_mods = {"BPSK", "QPSK", "16QAM"}
    if modulation not in valid_mods:
        raise HTTPException(
            status_code=422,
            detail=f"modulation must be one of {sorted(valid_mods)}",
        )

    # Size guard: 50 MB max
    MAX_BYTES = 50 * 1024 * 1024
    contents = await file.read()
    if len(contents) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit.")

    try:
        # 1. Parse IQ file
        signal = parse_iq_file(contents, file.filename or "upload.csv")

        # 2. Blind feature extraction
        feats = extract_features_blind(signal)

        # 3. Rule-based diagnosis
        diagnosis = diagnose_from_blind_features(feats, modulation)

        # 4. ML impairment detection (map blind features to model input)
        ml_features = {
            "evm_percent": feats["evm_est_percent"],
            "amp_std":     feats["amplitude_std"],
            "phase_std":   feats["phase_std"],
            "kurtosis":    feats["kurtosis"],
            "mean_amp_error": feats["amplitude_std"],
            "snr_db":      max(feats["snr_est_db"], 0),
            "freq_offset": abs(feats["freq_offset_est"]),
            "interference_power": max(feats["kurtosis"] * 0.1, 0),
            "modulation":  modulation,
        }
        imp_result = predict_impairment(ml_features)

        # 5. Build log message for failure categorizer
        log_msg = (
            f"[{'PASS' if imp_result['predicted_label'] == 1 else 'FAIL'}] {modulation} | "
            f"SNR_est={feats['snr_est_db']:.1f}dB | "
            f"FreqOffset_est={feats['freq_offset_est']:.4f} | "
            f"EVM_est={feats['evm_est_percent']:.2f}% | "
            f"PhaseStd={feats['phase_std']:.4f} | "
            f"Kurtosis={feats['kurtosis']:.3f} | "
            f"IQImbalance={feats['iq_imbalance']:.4f}"
        )

        # 6. Failure categorization from log text
        cat_result = categorize_failure(log_msg)
        category = cat_result["category"]

        return {
            "source": "real_iq_upload",
            "filename": file.filename,
            "modulation": modulation,
            "signal_info": {
                "n_samples": feats["n_samples"],
                "snr_est_db": feats["snr_est_db"],
                "freq_offset_est": feats["freq_offset_est"],
            },
            "blind_features": feats,
            "rule_based_diagnosis": diagnosis,
            "ml_impairment_detection": {
                "result": imp_result["label_name"],
                "pass_probability": round(imp_result["pass_probability"], 4),
                "fail_probability": round(imp_result["fail_probability"], 4),
            },
            "failure_category": category,
            "root_cause": ROOT_CAUSE_MAP.get(category, "Unknown"),
            "recommendations": RECOMMENDATIONS.get(category, []),
            "log_message": log_msg,
        }

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/triage/sample-iq", tags=["Real IQ Data"])
def download_sample_iq(
    modulation: str = "QPSK",
    snr_db: float = 15.0,
    n_symbols: int = 512,
):
    """
    Download a sample IQ CSV file to use as an upload template.
    Shows the expected format: two columns I and Q.
    """
    valid_mods = {"BPSK", "QPSK", "16QAM"}
    if modulation not in valid_mods:
        raise HTTPException(status_code=422, detail=f"modulation must be one of {sorted(valid_mods)}")
    csv_bytes = generate_sample_iq_csv(modulation=modulation, snr_db=snr_db, n_symbols=n_symbols)
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=sample_iq_{modulation}_{snr_db}dB.csv"},
    )
