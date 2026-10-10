"""
train_all.py
End-to-end pipeline script: generate data → train models → run triage.
Run this once to bootstrap the project.
"""

import os

print("=" * 60)
print("Step 1/4 — Generating signal dataset")
print("=" * 60)
os.makedirs("data", exist_ok=True)

from signal_generator import generate_dataset
df_signals = generate_dataset(n_samples=500)
df_signals.to_csv("data/signal_dataset.csv", index=False)
print(f"  Saved {len(df_signals)} rows → data/signal_dataset.csv\n")


print("=" * 60)
print("Step 2/4 — Generating test log")
print("=" * 60)
from test_cases import generate_test_log
df_log = generate_test_log(n_cases=300)
df_log.to_csv("data/test_log.csv", index=False)
print(f"  Saved {len(df_log)} rows → data/test_log.csv\n")


print("=" * 60)
print("Step 3/4 — Training RF impairment detector")
print("=" * 60)
from ml_impairment_detector import train as train_impairment
train_impairment()
print()


print("=" * 60)
print("Step 4/4 — Training failure categorizer")
print("=" * 60)
from ml_failure_categorizer import train as train_categorizer
train_categorizer()
print()


print("=" * 60)
print("Running batch triage to generate initial reports")
print("=" * 60)
import json
from triage_engine import triage_batch, build_summary_report
os.makedirs("outputs", exist_ok=True)
reports = triage_batch()
summary = build_summary_report(reports)
with open("outputs/summary_report.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n✓ All done. Reports saved to outputs/")
print(json.dumps(summary, indent=2))
