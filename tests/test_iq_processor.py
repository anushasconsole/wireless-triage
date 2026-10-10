"""
tests/test_iq_processor.py
Unit tests for real IQ file parsing and blind feature extraction.
"""

import io
import numpy as np
import pandas as pd
import pytest

from iq_processor import (
    parse_iq_file,
    extract_features_blind,
    diagnose_from_blind_features,
    generate_sample_iq_csv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_csv(i_vals, q_vals, header="I,Q") -> bytes:
    lines = [header] + [f"{i},{q}" for i, q in zip(i_vals, q_vals)]
    return "\n".join(lines).encode("utf-8")


def make_signal(n=256, snr=20.0) -> np.ndarray:
    sig = np.exp(1j * np.random.uniform(0, 2 * np.pi, n))
    noise_power = 1 / (10 ** (snr / 10))
    sig += np.sqrt(noise_power / 2) * (np.random.randn(n) + 1j * np.random.randn(n))
    return sig


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------

class TestParseIQFile:
    def test_csv_iq_columns(self):
        i = np.cos(np.linspace(0, 2 * np.pi, 64))
        q = np.sin(np.linspace(0, 2 * np.pi, 64))
        csv = make_csv(i, q, header="I,Q")
        sig = parse_iq_file(csv, "test.csv")
        assert len(sig) == 64
        assert np.iscomplexobj(sig)

    def test_csv_real_imag_columns(self):
        i = np.ones(32)
        q = np.zeros(32)
        csv = make_csv(i, q, header="real,imag")
        sig = parse_iq_file(csv, "test.csv")
        assert len(sig) == 32

    def test_npy_complex(self):
        arr = np.array([1 + 2j, 3 + 4j] * 20, dtype=np.complex128)
        buf = io.BytesIO()
        np.save(buf, arr)
        sig = parse_iq_file(buf.getvalue(), "test.npy")
        assert len(sig) == 40
        assert np.iscomplexobj(sig)

    def test_npy_2col_float(self):
        arr = np.column_stack([np.ones(32), np.zeros(32)]).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        sig = parse_iq_file(buf.getvalue(), "test.npy")
        assert len(sig) == 32

    def test_bin_float32(self):
        i = np.ones(32, dtype=np.float32)
        q = np.zeros(32, dtype=np.float32)
        interleaved = np.empty(64, dtype=np.float32)
        interleaved[0::2] = i
        interleaved[1::2] = q
        sig = parse_iq_file(interleaved.tobytes(), "test.bin")
        assert len(sig) == 32

    def test_txt_format(self):
        lines = "\n".join(f"{np.cos(t):.4f} {np.sin(t):.4f}" for t in np.linspace(0, 6, 50))
        sig = parse_iq_file(lines.encode("utf-8"), "test.txt")
        assert len(sig) == 50

    def test_unsupported_extension(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_iq_file(b"data", "test.wav")

    def test_too_short(self):
        csv = make_csv([1, 2], [0, 0])
        with pytest.raises(ValueError, match="too short"):
            parse_iq_file(csv, "test.csv")


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------

class TestExtractFeaturesBlind:
    def test_returns_all_keys(self):
        sig = make_signal(256)
        feats = extract_features_blind(sig)
        for key in [
            "n_samples", "amplitude_mean", "amplitude_std", "amplitude_range",
            "phase_std", "freq_offset_est", "kurtosis", "papr_db",
            "iq_imbalance", "evm_est_percent", "snr_est_db",
        ]:
            assert key in feats, f"Missing key: {key}"

    def test_n_samples_correct(self):
        sig = make_signal(128)
        feats = extract_features_blind(sig)
        assert feats["n_samples"] == 128

    def test_high_snr_low_evm(self):
        sig = make_signal(1024, snr=40.0)
        feats = extract_features_blind(sig)
        assert feats["evm_est_percent"] < 20.0

    def test_iq_imbalance_zero_for_balanced(self):
        # Perfectly balanced I/Q
        t = np.linspace(0, 2 * np.pi, 256)
        sig = np.cos(t) + 1j * np.sin(t)
        feats = extract_features_blind(sig)
        assert feats["iq_imbalance"] < 0.01


# ---------------------------------------------------------------------------
# Diagnosis tests
# ---------------------------------------------------------------------------

class TestDiagnoseBlind:
    def test_clean_signal_passes(self):
        sig = make_signal(1024, snr=30.0)
        feats = extract_features_blind(sig)
        # Force low kurtosis and phase std for a clean pass
        feats["evm_est_percent"] = 5.0
        feats["phase_std"] = 0.1
        feats["kurtosis"] = 0.3
        feats["iq_imbalance"] = 0.01
        feats["snr_est_db"] = 25.0
        result = diagnose_from_blind_features(feats, "QPSK")
        assert result["verdict"] == "PASS"

    def test_high_evm_detected(self):
        sig = make_signal(256)
        feats = extract_features_blind(sig)
        feats["evm_est_percent"] = 25.0  # force above QPSK threshold (15%)
        feats["phase_std"] = 0.1
        feats["kurtosis"] = 0.0
        feats["iq_imbalance"] = 0.0
        feats["snr_est_db"] = 20.0
        result = diagnose_from_blind_features(feats, "QPSK")
        assert "HIGH_EVM" in result["issues_detected"]


# ---------------------------------------------------------------------------
# Sample CSV generator test
# ---------------------------------------------------------------------------

def test_generate_sample_csv():
    csv_bytes = generate_sample_iq_csv("QPSK", snr_db=15.0, n_symbols=128)
    df = pd.read_csv(io.BytesIO(csv_bytes))
    assert "I" in df.columns
    assert "Q" in df.columns
    assert len(df) == 128
