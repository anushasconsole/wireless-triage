"""
signal_generator.py
Generates BPSK, QPSK, and 16-QAM signals with injected impairments:
  - Low SNR (AWGN noise)
  - Frequency offset
  - Interference (sinusoidal)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Modulation helpers
# ---------------------------------------------------------------------------

def generate_bpsk(n_symbols: int) -> np.ndarray:
    """Return BPSK constellation points (+1 or -1)."""
    bits = np.random.randint(0, 2, n_symbols)
    return 2 * bits - 1 + 0j


def generate_qpsk(n_symbols: int) -> np.ndarray:
    """Return QPSK constellation points."""
    bits = np.random.randint(0, 4, n_symbols)
    angles = np.pi / 4 + bits * np.pi / 2
    return np.exp(1j * angles)


def generate_16qam(n_symbols: int) -> np.ndarray:
    """Return 16-QAM constellation points (Gray-coded)."""
    levels = np.array([-3, -1, 1, 3])
    i = np.random.choice(levels, n_symbols)
    q = np.random.choice(levels, n_symbols)
    # Normalize average power to 1
    return (i + 1j * q) / np.sqrt(10)


MODULATORS = {
    "BPSK": generate_bpsk,
    "QPSK": generate_qpsk,
    "16QAM": generate_16qam,
}


# ---------------------------------------------------------------------------
# Impairment helpers
# ---------------------------------------------------------------------------

def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Additive White Gaussian Noise to reach the target SNR (dB)."""
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise


def add_frequency_offset(signal: np.ndarray, freq_offset: float, sample_rate: float = 1.0) -> np.ndarray:
    """Apply a frequency offset (Hz) to the signal."""
    t = np.arange(len(signal)) / sample_rate
    return signal * np.exp(1j * 2 * np.pi * freq_offset * t)


def add_interference(signal: np.ndarray, interference_power: float, freq: float = 0.1) -> np.ndarray:
    """Inject a sinusoidal interferer."""
    t = np.arange(len(signal))
    interferer = np.sqrt(interference_power) * np.exp(1j * 2 * np.pi * freq * t)
    return signal + interferer


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(received: np.ndarray, reference: np.ndarray) -> dict:
    """
    Extract signal-quality features from a (received, reference) pair.

    Returns a dict with EVM (%), magnitude std, phase std, kurtosis of
    amplitude, and mean amplitude error.
    """
    # EVM
    error = received - reference
    evm_rms = np.sqrt(np.mean(np.abs(error) ** 2)) / np.sqrt(np.mean(np.abs(reference) ** 2))
    evm_percent = evm_rms * 100

    # Amplitude and phase
    amp_received = np.abs(received)
    amp_ref = np.abs(reference)
    phase_diff = np.angle(received) - np.angle(reference)

    # Kurtosis of amplitude (excess)
    amp_mean = np.mean(amp_received)
    amp_std = np.std(amp_received)
    kurtosis = np.mean((amp_received - amp_mean) ** 4) / (amp_std ** 4) - 3 if amp_std > 0 else 0

    return {
        "evm_percent": evm_percent,
        "amp_std": float(np.std(amp_received - amp_ref)),
        "phase_std": float(np.std(phase_diff)),
        "kurtosis": float(kurtosis),
        "mean_amp_error": float(np.mean(np.abs(amp_received - amp_ref))),
    }


# ---------------------------------------------------------------------------
# BER estimation (Monte-Carlo approximation)
# ---------------------------------------------------------------------------

def estimate_ber(snr_db: float, modulation: str) -> float:
    """Rough theoretical BER for AWGN channel."""
    snr_lin = 10 ** (snr_db / 10)
    if modulation == "BPSK":
        from scipy.special import erfc
        return 0.5 * erfc(np.sqrt(snr_lin))
    elif modulation == "QPSK":
        from scipy.special import erfc
        return 0.5 * erfc(np.sqrt(snr_lin / 2))
    else:  # 16-QAM approximation
        from scipy.special import erfc
        return (3 / 8) * erfc(np.sqrt(snr_lin / 10))


# ---------------------------------------------------------------------------
# Main dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    n_samples: int = 500,
    n_symbols: int = 256,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a labelled dataset of signal-feature rows with pass/fail labels.

    Each row represents one test case with randomly chosen modulation,
    SNR, frequency offset, and interference level.
    """
    rng = np.random.default_rng(seed)
    records = []

    modulations = list(MODULATORS.keys())

    for _ in range(n_samples):
        mod = rng.choice(modulations)
        snr_db = rng.uniform(5, 30)
        freq_offset = rng.uniform(0, 0.05)
        interference_power = rng.uniform(0, 0.3)

        # Generate clean reference
        reference = MODULATORS[mod](n_symbols)

        # Apply impairments
        received = add_awgn(reference, snr_db)
        received = add_frequency_offset(received, freq_offset)
        received = add_interference(received, interference_power)

        # Extract features
        feats = extract_features(received, reference)
        ber = estimate_ber(snr_db, mod)

        # Pass/fail thresholds
        evm_threshold = {"BPSK": 20.0, "QPSK": 15.0, "16QAM": 10.0}[mod]
        ber_threshold = {"BPSK": 0.01, "QPSK": 0.01, "16QAM": 0.05}[mod]
        passed = int(feats["evm_percent"] <= evm_threshold and ber <= ber_threshold)

        records.append(
            {
                "modulation": mod,
                "snr_db": snr_db,
                "freq_offset": freq_offset,
                "interference_power": interference_power,
                "ber": ber,
                **feats,
                "label": passed,  # 1 = PASS, 0 = FAIL
            }
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_dataset(n_samples=500)
    df.to_csv("data/signal_dataset.csv", index=False)
    print(f"Generated {len(df)} samples. PASS rate: {df['label'].mean():.1%}")
    print(df.head())
