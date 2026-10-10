"""
iq_processor.py
Handles real IQ data uploaded by users.

Supported file formats:
  - CSV:  two columns  (I, Q)  or one complex column
  - NPY:  numpy array of complex64/complex128
  - BIN:  raw interleaved float32  [I0, Q0, I1, Q1, ...]
  - TXT:  whitespace-separated I Q pairs

Because we have no clean reference signal (this is real captured data),
features are extracted blind using statistical properties of the IQ stream
rather than error-vs-reference.
"""

import io
import numpy as np
import pandas as pd
from typing import Union


# ---------------------------------------------------------------------------
# File parsers
# ---------------------------------------------------------------------------

def _parse_csv(data: bytes) -> np.ndarray:
    """
    Accept CSV with columns:
      - 'I','Q'  (two real columns)
      - 'real','imag'
      - single column of complex strings like '0.5+0.3j'
    """
    df = pd.read_csv(io.BytesIO(data))
    df.columns = [c.strip().lower() for c in df.columns]

    if "i" in df.columns and "q" in df.columns:
        return df["i"].values + 1j * df["q"].values
    if "real" in df.columns and "imag" in df.columns:
        return df["real"].values + 1j * df["imag"].values
    if len(df.columns) == 2:
        return df.iloc[:, 0].values + 1j * df.iloc[:, 1].values
    if len(df.columns) == 1:
        return df.iloc[:, 0].apply(complex).values

    raise ValueError(
        "CSV must have columns: I,Q  or  real,imag  or two unnamed numeric columns."
    )


def _parse_npy(data: bytes) -> np.ndarray:
    arr = np.load(io.BytesIO(data), allow_pickle=False)
    if not np.iscomplexobj(arr):
        if arr.ndim == 2 and arr.shape[1] == 2:
            arr = arr[:, 0] + 1j * arr[:, 1]
        else:
            raise ValueError("NPY file must contain complex data or shape (N,2) float.")
    return arr.flatten()


def _parse_bin(data: bytes) -> np.ndarray:
    """Interleaved float32: I0 Q0 I1 Q1 ..."""
    floats = np.frombuffer(data, dtype=np.float32)
    if len(floats) % 2 != 0:
        floats = floats[:-1]
    return floats[0::2] + 1j * floats[1::2]


def _parse_txt(data: bytes) -> np.ndarray:
    """Whitespace-separated I Q pairs, one per line."""
    lines = data.decode("utf-8").strip().splitlines()
    pairs = [line.split() for line in lines if line.strip()]
    arr = np.array([(float(p[0]), float(p[1])) for p in pairs if len(p) >= 2])
    return arr[:, 0] + 1j * arr[:, 1]


PARSERS = {
    "csv": _parse_csv,
    "npy": _parse_npy,
    "bin": _parse_bin,
    "txt": _parse_txt,
}


def parse_iq_file(data: bytes, filename: str) -> np.ndarray:
    """Detect format from extension and parse into complex numpy array."""
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in PARSERS:
        raise ValueError(f"Unsupported file type '.{ext}'. Supported: {list(PARSERS.keys())}")
    signal = PARSERS[ext](data)
    if len(signal) < 16:
        raise ValueError("File contains fewer than 16 IQ samples — too short to analyse.")
    return signal.astype(np.complex128)


# ---------------------------------------------------------------------------
# Blind feature extraction (no reference signal)
# ---------------------------------------------------------------------------

def extract_features_blind(signal: np.ndarray) -> dict:
    """
    Extract signal quality features from captured IQ data without a reference.

    Uses statistical properties that correlate with impairment types:
      - amplitude_mean / std / range  → gain issues, noise floor
      - phase_std                      → phase noise, freq offset
      - kurtosis                       → interference spikes vs Gaussian noise
      - papr                           → peak-to-average power ratio
      - iq_imbalance                   → I vs Q amplitude mismatch
      - evm_est                        → blind EVM estimate via cluster variance
    """
    amp = np.abs(signal)
    phase = np.angle(signal)

    # Amplitude stats
    amp_mean = float(np.mean(amp))
    amp_std = float(np.std(amp))
    amp_range = float(np.max(amp) - np.min(amp))

    # Phase stats
    phase_diff = np.diff(np.unwrap(phase))
    phase_std = float(np.std(phase_diff))
    freq_offset_est = float(np.mean(phase_diff) / (2 * np.pi))  # normalised

    # Kurtosis of amplitude (excess kurtosis; Gaussian=0, spiky>0)
    if amp_std > 0:
        kurtosis = float(np.mean((amp - amp_mean) ** 4) / amp_std ** 4 - 3)
    else:
        kurtosis = 0.0

    # Peak-to-average power ratio
    power = amp ** 2
    papr = float(np.max(power) / np.mean(power)) if np.mean(power) > 0 else 0.0

    # IQ imbalance: I vs Q amplitude ratio (ideal = 1.0)
    i_power = float(np.mean(signal.real ** 2))
    q_power = float(np.mean(signal.imag ** 2))
    iq_imbalance = float(abs(i_power - q_power) / (i_power + q_power + 1e-12))

    # Blind EVM estimate: normalise signal, measure cluster spread
    norm_signal = signal / (amp_mean + 1e-12)
    evm_est = float(np.std(np.abs(norm_signal) - 1.0) * 100)

    # SNR estimate from amplitude distribution (moment method)
    snr_est_db = float(10 * np.log10(amp_mean ** 2 / (amp_std ** 2 + 1e-12)))

    return {
        "n_samples": len(signal),
        "amplitude_mean": round(amp_mean, 6),
        "amplitude_std": round(amp_std, 6),
        "amplitude_range": round(amp_range, 6),
        "phase_std": round(phase_std, 6),
        "freq_offset_est": round(freq_offset_est, 6),
        "kurtosis": round(kurtosis, 4),
        "papr_db": round(10 * np.log10(papr + 1e-12), 4),
        "iq_imbalance": round(iq_imbalance, 6),
        "evm_est_percent": round(evm_est, 4),
        "snr_est_db": round(snr_est_db, 4),
    }


# ---------------------------------------------------------------------------
# Impairment diagnosis from blind features (rule-based + threshold)
# ---------------------------------------------------------------------------

def diagnose_from_blind_features(feats: dict, modulation: str) -> dict:
    """
    Rule-based diagnosis when no ML model reference is available.
    Complements the ML model by providing a physics-based cross-check.
    """
    evm_thresholds = {"BPSK": 20.0, "QPSK": 15.0, "16QAM": 10.0}
    evm_thresh = evm_thresholds.get(modulation, 15.0)

    issues = []
    details = []

    if feats["evm_est_percent"] > evm_thresh:
        issues.append("HIGH_EVM")
        details.append(f"Blind EVM estimate {feats['evm_est_percent']:.1f}% exceeds {evm_thresh}% threshold")

    if feats["phase_std"] > 0.3:
        issues.append("PHASE_NOISE")
        details.append(f"Phase std {feats['phase_std']:.4f} rad exceeds 0.3 rad threshold")

    if feats["kurtosis"] > 1.5:
        issues.append("INTERFERENCE")
        details.append(f"Kurtosis {feats['kurtosis']:.2f} indicates non-Gaussian interference")

    if feats["iq_imbalance"] > 0.05:
        issues.append("IQ_IMBALANCE")
        details.append(f"IQ imbalance {feats['iq_imbalance']*100:.1f}% — I/Q power mismatch detected")

    if feats["snr_est_db"] < 10:
        issues.append("LOW_SNR")
        details.append(f"Estimated SNR {feats['snr_est_db']:.1f} dB below 10 dB minimum")

    if not issues:
        verdict = "PASS"
        summary = "Signal appears healthy. All blind metrics within normal range."
    elif len(issues) == 1:
        verdict = issues[0]
        summary = details[0]
    else:
        verdict = "MULTIPLE"
        summary = " | ".join(details)

    return {
        "verdict": verdict,
        "issues_detected": issues,
        "detail": summary,
    }


# ---------------------------------------------------------------------------
# Sample file generator (for users to download as a template)
# ---------------------------------------------------------------------------

def generate_sample_iq_csv(
    modulation: str = "QPSK",
    n_symbols: int = 512,
    snr_db: float = 15.0,
    seed: int = 1,
) -> bytes:
    """Generate a sample IQ CSV file users can download as a template."""
    from signal_generator import MODULATORS, add_awgn, add_frequency_offset

    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    signal = MODULATORS[modulation](n_symbols)
    signal = add_awgn(signal, snr_db)
    signal = add_frequency_offset(signal, 0.01)

    df = pd.DataFrame({"I": signal.real, "Q": signal.imag})
    return df.to_csv(index=False).encode("utf-8")
