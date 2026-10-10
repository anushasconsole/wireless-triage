"""
api.py
FastAPI application that exposes the wireless triage engine via REST endpoints.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import json
import os

from triage_engine import triage_case, triage_batch, build_summary_report
from ml_impairment_detector import predict_impairment
from ml_failure_categorizer import categorize_failure


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
