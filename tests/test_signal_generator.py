"""
tests/test_signal_generator.py
Unit tests for signal generation and feature extraction.
"""

import numpy as np
import pytest

from signal_generator import (
    generate_bpsk,
    generate_qpsk,
    generate_16qam,
    add_awgn,
    add_frequency_offset,
    add_interference,
    extract_features,
    estimate_ber,
    generate_dataset,
)


N = 256  # symbols per test


class TestModulators:
    def test_bpsk_values(self):
        sig = generate_bpsk(N)
        assert len(sig) == N
        # BPSK: real part only, ±1
        assert np.allclose(np.abs(sig), 1.0)
        assert np.allclose(sig.imag, 0.0)

    def test_qpsk_unit_circle(self):
        sig = generate_qpsk(N)
        assert len(sig) == N
        assert np.allclose(np.abs(sig), 1.0, atol=1e-10)

    def test_16qam_shape(self):
        sig = generate_16qam(N)
        assert len(sig) == N
        # Normalized: mean power ≈ 1
        power = np.mean(np.abs(sig) ** 2)
        assert abs(power - 1.0) < 0.1


class TestImpairments:
    def test_awgn_increases_noise(self):
        ref = generate_bpsk(1024)
        noisy = add_awgn(ref, snr_db=10)
        error_power = np.mean(np.abs(noisy - ref) ** 2)
        assert error_power > 0

    def test_high_snr_low_noise(self):
        ref = generate_bpsk(1024)
        noisy = add_awgn(ref, snr_db=40)
        evm = np.sqrt(np.mean(np.abs(noisy - ref) ** 2)) / np.sqrt(np.mean(np.abs(ref) ** 2))
        assert evm < 0.02  # < 2% EVM at 40 dB SNR

    def test_freq_offset_rotates_phase(self):
        ref = generate_qpsk(N)
        shifted = add_frequency_offset(ref, freq_offset=0.1)
        # Phase should differ
        phase_diff = np.angle(shifted[1]) - np.angle(ref[1])
        assert abs(phase_diff) > 0

    def test_interference_adds_power(self):
        ref = generate_bpsk(N)
        interfered = add_interference(ref, interference_power=1.0)
        assert np.mean(np.abs(interfered) ** 2) > np.mean(np.abs(ref) ** 2)


class TestFeatureExtraction:
    def test_clean_signal_low_evm(self):
        ref = generate_bpsk(N)
        feats = extract_features(ref, ref)
        assert feats["evm_percent"] == pytest.approx(0.0, abs=1e-10)

    def test_evm_increases_with_noise(self):
        ref = generate_qpsk(1024)
        noisy = add_awgn(ref, snr_db=10)
        feats = extract_features(noisy, ref)
        assert feats["evm_percent"] > 0

    def test_feature_keys(self):
        ref = generate_16qam(N)
        noisy = add_awgn(ref, 20)
        feats = extract_features(noisy, ref)
        for key in ["evm_percent", "amp_std", "phase_std", "kurtosis", "mean_amp_error"]:
            assert key in feats


class TestBER:
    def test_ber_decreases_with_snr(self):
        ber_low = estimate_ber(5, "BPSK")
        ber_high = estimate_ber(20, "BPSK")
        assert ber_low > ber_high

    def test_ber_positive(self):
        for mod in ["BPSK", "QPSK", "16QAM"]:
            assert estimate_ber(15, mod) >= 0


class TestDatasetGeneration:
    def test_dataset_shape(self):
        df = generate_dataset(n_samples=50, n_symbols=64)
        assert len(df) == 50
        assert "label" in df.columns

    def test_labels_binary(self):
        df = generate_dataset(n_samples=50, n_symbols=64)
        assert set(df["label"].unique()).issubset({0, 1})
