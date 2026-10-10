"""
test_cases.py
Runs pass/fail test cases against generated signals and writes a test log CSV.
Each row in the log describes a test case outcome with a human-readable message.
"""

import numpy as np
import pandas as pd

from signal_generator import (
    MODULATORS,
    add_awgn,
    add_frequency_offset,
    add_interference,
    extract_features,
    estimate_ber,
)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

EVM_THRESHOLDS = {"BPSK": 20.0, "QPSK": 15.0, "16QAM": 10.0}
BER_THRESHOLDS = {"BPSK": 0.01, "QPSK": 0.01, "16QAM": 0.05}


# ---------------------------------------------------------------------------
# Individual test evaluators
# ---------------------------------------------------------------------------

def check_evm(evm_percent: float, threshold: float) -> tuple[bool, str]:
    passed = evm_percent <= threshold
    msg = (
        f"EVM {evm_percent:.2f}% within threshold {threshold}%"
        if passed
        else f"EVM {evm_percent:.2f}% exceeds threshold {threshold}%"
    )
    return passed, msg


def check_ber(ber: float, threshold: float) -> tuple[bool, str]:
    passed = ber <= threshold
    msg = (
        f"BER {ber:.4e} within threshold {threshold:.4e}"
        if passed
        else f"BER {ber:.4e} exceeds threshold {threshold:.4e}"
    )
    return passed, msg


def check_phase_noise(phase_std: float, threshold: float = 0.3) -> tuple[bool, str]:
    passed = phase_std <= threshold
    msg = (
        f"Phase noise {phase_std:.4f} rad within threshold"
        if passed
        else f"Phase noise {phase_std:.4f} rad exceeds threshold {threshold} rad"
    )
    return passed, msg


# ---------------------------------------------------------------------------
# Test case runner
# ---------------------------------------------------------------------------

def run_test_case(
    mod: str,
    snr_db: float,
    freq_offset: float,
    interference_power: float,
    n_symbols: int = 256,
    case_id: int = 0,
) -> dict:
    """Run a single test case and return a log record."""
    reference = MODULATORS[mod](n_symbols)
    received = add_awgn(reference, snr_db)
    received = add_frequency_offset(received, freq_offset)
    received = add_interference(received, interference_power)

    feats = extract_features(received, reference)
    ber = estimate_ber(snr_db, mod)

    evm_pass, evm_msg = check_evm(feats["evm_percent"], EVM_THRESHOLDS[mod])
    ber_pass, ber_msg = check_ber(ber, BER_THRESHOLDS[mod])
    phase_pass, phase_msg = check_phase_noise(feats["phase_std"])

    overall = evm_pass and ber_pass and phase_pass

    # Build failure reason string for the log
    reasons = []
    if not evm_pass:
        reasons.append("HIGH_EVM")
    if not ber_pass:
        reasons.append("HIGH_BER")
    if not phase_pass:
        reasons.append("PHASE_NOISE")

    failure_reason = "; ".join(reasons) if reasons else "NONE"

    log_message = (
        f"[{'PASS' if overall else 'FAIL'}] {mod} | "
        f"SNR={snr_db:.1f}dB | "
        f"FreqOffset={freq_offset:.4f} | "
        f"Interference={interference_power:.3f} | "
        f"{evm_msg} | {ber_msg} | {phase_msg}"
    )

    return {
        "case_id": case_id,
        "modulation": mod,
        "snr_db": snr_db,
        "freq_offset": freq_offset,
        "interference_power": interference_power,
        "evm_percent": feats["evm_percent"],
        "ber": ber,
        "phase_std": feats["phase_std"],
        "evm_pass": int(evm_pass),
        "ber_pass": int(ber_pass),
        "phase_pass": int(phase_pass),
        "overall_pass": int(overall),
        "failure_reason": failure_reason,
        "log_message": log_message,
    }


def generate_test_log(
    n_cases: int = 300,
    n_symbols: int = 256,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a test log with n_cases test cases across all modulations."""
    rng = np.random.default_rng(seed)
    modulations = list(MODULATORS.keys())
    records = []

    for i in range(n_cases):
        mod = rng.choice(modulations)
        snr_db = rng.uniform(5, 30)
        freq_offset = rng.uniform(0, 0.05)
        interference_power = rng.uniform(0, 0.3)

        record = run_test_case(
            mod, snr_db, freq_offset, interference_power, n_symbols, case_id=i
        )
        records.append(record)
        print(record["log_message"])

    return pd.DataFrame(records)


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    df = generate_test_log(n_cases=300)
    df.to_csv("data/test_log.csv", index=False)
    pass_rate = df["overall_pass"].mean()
    print(f"\nTotal cases: {len(df)} | PASS rate: {pass_rate:.1%}")
    print(df["failure_reason"].value_counts())
