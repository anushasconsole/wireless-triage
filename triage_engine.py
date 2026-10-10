"""
triage_engine.py
Automated root-cause triage engine.

Combines:
  1. Impairment Detector  (RF features → PASS/FAIL)
  2. Failure Categorizer  (log message → failure category)

Produces a structured summary report for each test case.
"""

import os
import json
import pandas as pd
from datetime import datetime, timezone

from signal_generator import (
    MODULATORS,
    add_awgn,
    add_frequency_offset,
    add_interference,
    extract_features,
    estimate_ber,
)
from ml_impairment_detector import predict_impairment
from ml_failure_categorizer import categorize_failure


# ---------------------------------------------------------------------------
# Root-cause mapping
# ---------------------------------------------------------------------------

ROOT_CAUSE_MAP = {
    "HIGH_EVM": (
        "High Error Vector Magnitude detected. Likely causes: excessive noise, "
        "IQ imbalance, non-linearity, or large frequency offset."
    ),
    "HIGH_BER": (
        "Bit Error Rate exceeds threshold. Likely causes: low SNR, multipath "
        "fading, or strong interference."
    ),
    "PHASE_NOISE": (
        "Phase noise / phase deviation exceeds limit. Likely causes: oscillator "
        "instability, large frequency offset, or Doppler shift."
    ),
    "MULTIPLE": (
        "Multiple failure modes detected simultaneously. Inspect EVM, BER, and "
        "phase metrics together; may indicate severe channel degradation."
    ),
    "PASS": "All metrics within specification. No corrective action required.",
}

RECOMMENDATIONS = {
    "HIGH_EVM": [
        "Increase transmit power or reduce path loss.",
        "Calibrate IQ imbalance on the transmitter.",
        "Apply frequency offset correction (AFC).",
    ],
    "HIGH_BER": [
        "Increase link margin / SNR.",
        "Apply forward error correction (FEC) coding.",
        "Mitigate interference source.",
    ],
    "PHASE_NOISE": [
        "Replace or tune the reference oscillator.",
        "Apply phase-locked loop (PLL) correction.",
        "Reduce frequency offset with AFC loop.",
    ],
    "MULTIPLE": [
        "Perform end-to-end link budget analysis.",
        "Check hardware health (PA, LNA, mixer).",
        "Review environmental RF interference.",
    ],
    "PASS": ["Continue normal operation."],
}


# ---------------------------------------------------------------------------
# Per-case triage
# ---------------------------------------------------------------------------

def triage_case(
    modulation: str,
    snr_db: float,
    freq_offset: float,
    interference_power: float,
    case_id: int = 0,
    n_symbols: int = 256,
) -> dict:
    """
    Run a complete triage on a single test configuration.

    Returns a structured report dict.
    """
    # 1. Simulate signal
    reference = MODULATORS[modulation](n_symbols)
    received = add_awgn(reference, snr_db)
    received = add_frequency_offset(received, freq_offset)
    received = add_interference(received, interference_power)

    feats = extract_features(received, reference)
    ber = estimate_ber(snr_db, modulation)

    # 2. Impairment detection
    imp_result = predict_impairment(
        {
            **feats,
            "snr_db": snr_db,
            "freq_offset": freq_offset,
            "interference_power": interference_power,
            "modulation": modulation,
        }
    )

    # 3. Build log message for failure categorizer
    log_msg = (
        f"[{'PASS' if imp_result['predicted_label'] == 1 else 'FAIL'}] {modulation} | "
        f"SNR={snr_db:.1f}dB | FreqOffset={freq_offset:.4f} | "
        f"Interference={interference_power:.3f} | "
        f"EVM={feats['evm_percent']:.2f}% | BER={ber:.4e} | "
        f"PhaseStd={feats['phase_std']:.4f}"
    )

    # 4. Failure categorization
    cat_result = categorize_failure(log_msg)
    category = cat_result["category"]

    # 5. Compose report
    report = {
        "case_id": case_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modulation": modulation,
        "parameters": {
            "snr_db": snr_db,
            "freq_offset_hz": freq_offset,
            "interference_power": interference_power,
        },
        "metrics": {
            "evm_percent": round(feats["evm_percent"], 4),
            "ber": float(f"{ber:.6e}"),
            "phase_std_rad": round(feats["phase_std"], 6),
            "kurtosis": round(feats["kurtosis"], 4),
        },
        "impairment_detection": {
            "result": imp_result["label_name"],
            "pass_probability": round(imp_result["pass_probability"], 4),
            "fail_probability": round(imp_result["fail_probability"], 4),
        },
        "failure_category": category,
        "root_cause": ROOT_CAUSE_MAP.get(category, "Unknown"),
        "recommendations": RECOMMENDATIONS.get(category, []),
        "log_message": log_msg,
    }

    return report


# ---------------------------------------------------------------------------
# Batch triage
# ---------------------------------------------------------------------------

def triage_batch(
    test_log_path: str = "data/test_log.csv",
    output_path: str = "outputs/triage_report.json",
) -> list[dict]:
    """
    Run triage over every row in a test log CSV and write a JSON report.
    """
    df = pd.read_csv(test_log_path)
    reports = []

    for _, row in df.iterrows():
        report = triage_case(
            modulation=row["modulation"],
            snr_db=row["snr_db"],
            freq_offset=row["freq_offset"],
            interference_power=row["interference_power"],
            case_id=int(row["case_id"]),
        )
        reports.append(report)

    os.makedirs("outputs", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(reports, f, indent=2)

    total = len(reports)
    passes = sum(1 for r in reports if r["failure_category"] == "PASS")
    print(f"Triage complete: {total} cases | PASS: {passes} ({passes/total:.1%})")

    category_counts: dict = {}
    for r in reports:
        cat = r["failure_category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    print("Category breakdown:", json.dumps(category_counts, indent=2))

    return reports


# ---------------------------------------------------------------------------
# Summary report (human-readable)
# ---------------------------------------------------------------------------

def build_summary_report(reports: list[dict]) -> dict:
    """Aggregate triage results into a high-level summary."""
    total = len(reports)
    category_counts: dict = {}
    for r in reports:
        cat = r["failure_category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    avg_evm = sum(r["metrics"]["evm_percent"] for r in reports) / total
    avg_ber = sum(r["metrics"]["ber"] for r in reports) / total
    pass_rate = category_counts.get("PASS", 0) / total

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": total,
        "pass_rate": round(pass_rate, 4),
        "average_evm_percent": round(avg_evm, 4),
        "average_ber": float(f"{avg_ber:.4e}"),
        "failure_category_breakdown": category_counts,
        "top_recommendations": RECOMMENDATIONS.get(
            max(
                (k for k in category_counts if k != "PASS"),
                key=lambda k: category_counts[k],
                default="PASS",
            ),
            [],
        ),
    }


if __name__ == "__main__":
    reports = triage_batch()
    summary = build_summary_report(reports)
    print("\n=== Summary Report ===")
    print(json.dumps(summary, indent=2))

    with open("outputs/summary_report.json", "w") as f:
        json.dump(summary, f, indent=2)
