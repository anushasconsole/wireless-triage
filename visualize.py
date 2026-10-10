"""
visualize.py
Generates exploratory plots for the signal dataset and test log.
Outputs saved to outputs/plots/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from signal_generator import MODULATORS, add_awgn, add_frequency_offset, add_interference

os.makedirs("outputs/plots", exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Constellation diagrams
# ---------------------------------------------------------------------------

def plot_constellations():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    mods = [("BPSK", 10), ("QPSK", 15), ("16QAM", 20)]

    for ax, (mod, snr) in zip(axes, mods):
        ref = MODULATORS[mod](512)
        rx = add_awgn(ref, snr)
        rx = add_frequency_offset(rx, 0.01)

        ax.scatter(rx.real, rx.imag, s=4, alpha=0.5, label="Received")
        ax.scatter(ref.real, ref.imag, s=30, marker="x", c="red", label="Ideal")
        ax.set_title(f"{mod} @ SNR={snr} dB")
        ax.set_xlabel("In-phase")
        ax.set_ylabel("Quadrature")
        ax.legend(fontsize=7)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Constellation Diagrams with AWGN + Frequency Offset", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/plots/constellations.png", dpi=150)
    plt.close()
    print("Saved: outputs/plots/constellations.png")


# ---------------------------------------------------------------------------
# 2. EVM vs SNR curves
# ---------------------------------------------------------------------------

def plot_evm_vs_snr():
    snr_range = np.linspace(5, 30, 50)
    fig, ax = plt.subplots(figsize=(7, 4))

    for mod in ["BPSK", "QPSK", "16QAM"]:
        evms = []
        for snr in snr_range:
            ref = MODULATORS[mod](512)
            rx = add_awgn(ref, snr)
            error = rx - ref
            evm = np.sqrt(np.mean(np.abs(error) ** 2)) / np.sqrt(np.mean(np.abs(ref) ** 2)) * 100
            evms.append(evm)
        ax.plot(snr_range, evms, label=mod)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("EVM (%)")
    ax.set_title("EVM vs SNR (AWGN only)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/plots/evm_vs_snr.png", dpi=150)
    plt.close()
    print("Saved: outputs/plots/evm_vs_snr.png")


# ---------------------------------------------------------------------------
# 3. Dataset distribution
# ---------------------------------------------------------------------------

def plot_dataset_distributions(data_path: str = "data/signal_dataset.csv"):
    if not os.path.exists(data_path):
        print(f"  {data_path} not found — skipping dataset distribution plot.")
        return

    df = pd.read_csv(data_path)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # SNR distribution by label
    for label, name in [(0, "FAIL"), (1, "PASS")]:
        axes[0, 0].hist(df[df["label"] == label]["snr_db"], bins=20, alpha=0.6, label=name)
    axes[0, 0].set_xlabel("SNR (dB)")
    axes[0, 0].set_title("SNR Distribution by Label")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # EVM distribution
    for label, name in [(0, "FAIL"), (1, "PASS")]:
        axes[0, 1].hist(df[df["label"] == label]["evm_percent"], bins=20, alpha=0.6, label=name)
    axes[0, 1].set_xlabel("EVM (%)")
    axes[0, 1].set_title("EVM Distribution by Label")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Modulation counts
    mod_counts = df.groupby(["modulation", "label"]).size().unstack(fill_value=0)
    mod_counts.plot(kind="bar", ax=axes[0, 2], rot=0)
    axes[0, 2].set_title("Sample Count by Modulation & Label")
    axes[0, 2].set_xlabel("Modulation")
    axes[0, 2].grid(True, alpha=0.3)

    # Interference vs EVM scatter
    axes[1, 0].scatter(df["interference_power"], df["evm_percent"], s=4, alpha=0.3)
    axes[1, 0].set_xlabel("Interference Power")
    axes[1, 0].set_ylabel("EVM (%)")
    axes[1, 0].set_title("Interference Power vs EVM")
    axes[1, 0].grid(True, alpha=0.3)

    # Freq offset vs phase std
    axes[1, 1].scatter(df["freq_offset"], df["phase_std"], s=4, alpha=0.3)
    axes[1, 1].set_xlabel("Frequency Offset")
    axes[1, 1].set_ylabel("Phase Std (rad)")
    axes[1, 1].set_title("Freq Offset vs Phase Noise")
    axes[1, 1].grid(True, alpha=0.3)

    # BER distribution (log scale)
    axes[1, 2].hist(np.log10(df["ber"].clip(lower=1e-10)), bins=30, color="steelblue")
    axes[1, 2].set_xlabel("log10(BER)")
    axes[1, 2].set_title("BER Distribution")
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle("Signal Dataset Distributions", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/plots/dataset_distributions.png", dpi=150)
    plt.close()
    print("Saved: outputs/plots/dataset_distributions.png")


# ---------------------------------------------------------------------------
# 4. Test log failure breakdown
# ---------------------------------------------------------------------------

def plot_test_log_summary(log_path: str = "data/test_log.csv"):
    if not os.path.exists(log_path):
        print(f"  {log_path} not found — skipping test log plot.")
        return

    df = pd.read_csv(log_path)

    # Map to primary category
    from ml_failure_categorizer import encode_failure_category
    df["category"] = df["failure_reason"].apply(encode_failure_category)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Category pie chart
    counts = df["category"].value_counts()
    axes[0].pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140)
    axes[0].set_title("Failure Category Breakdown")

    # Pass rate per modulation
    pass_rate = df.groupby("modulation")["overall_pass"].mean().sort_values()
    pass_rate.plot(kind="barh", ax=axes[1], color="steelblue")
    axes[1].set_xlabel("Pass Rate")
    axes[1].set_title("Pass Rate by Modulation")
    axes[1].axvline(0.5, color="red", linestyle="--", alpha=0.7)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Test Log Summary", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/plots/test_log_summary.png", dpi=150)
    plt.close()
    print("Saved: outputs/plots/test_log_summary.png")


if __name__ == "__main__":
    plot_constellations()
    plot_evm_vs_snr()
    plot_dataset_distributions()
    plot_test_log_summary()
    print("\nAll plots saved to outputs/plots/")
