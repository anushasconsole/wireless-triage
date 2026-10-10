"""
tests/test_api.py
Integration tests for the FastAPI application.
Models must be trained (run train_all.py) before these tests pass.
"""

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.parametrize("modulation", ["BPSK", "QPSK", "16QAM"])
def test_triage_case_valid(modulation):
    payload = {
        "modulation": modulation,
        "snr_db": 15.0,
        "freq_offset": 0.01,
        "interference_power": 0.05,
        "case_id": 0,
    }
    response = client.post("/triage/case", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert "failure_category" in data
    assert "root_cause" in data
    assert "recommendations" in data
    assert "metrics" in data
    assert "impairment_detection" in data


def test_triage_case_invalid_modulation():
    payload = {
        "modulation": "AM",
        "snr_db": 15.0,
        "freq_offset": 0.01,
        "interference_power": 0.05,
    }
    response = client.post("/triage/case", json=payload)
    assert response.status_code == 422


def test_predict_impairment():
    payload = {
        "modulation": "QPSK",
        "snr_db": 12.0,
        "freq_offset": 0.02,
        "interference_power": 0.1,
        "evm_percent": 8.5,
        "amp_std": 0.1,
        "phase_std": 0.15,
        "kurtosis": 0.0,
        "mean_amp_error": 0.05,
    }
    response = client.post("/predict/impairment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "label_name" in data
    assert data["label_name"] in {"PASS", "FAIL"}


def test_predict_category():
    payload = {
        "log_message": (
            "[FAIL] QPSK | SNR=8.0dB | FreqOffset=0.0300 | "
            "Interference=0.200 | EVM 18.50% exceeds threshold 15% | "
            "BER 2.5000e-02 exceeds threshold 1.0000e-02 | "
            "Phase noise 0.2100 rad within threshold"
        )
    }
    response = client.post("/predict/category", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "probabilities" in data
