"""
ui.py
Streamlit UI for the Wireless Triage Pipeline.

Two modes:
  Tab 1 — Upload real IQ data (.csv/.npy/.bin/.txt)
  Tab 2 — Simulate a signal with custom parameters

Run with:
  streamlit run ui.py
"""

import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Local imports (runs in same process as the models)
from iq_processor import (
    parse_iq_file,
    extract_features_blind,
    diagnose_from_blind_features,
    generate_sample_iq_csv,
)
from triage_engine import triage_case, ROOT_CAUSE_MAP, RECOMMENDATIONS
from ml_impairment_detector import predict_impairment
from ml_failure_categorizer import categorize_failure
from signal_generator import MODULATORS, add_awgn, add_frequency_offset, add_interference


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Wireless Signal Triage",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "PASS":        "#2ecc71",
    "HIGH_EVM":    "#e74c3c",
    "HIGH_BER":    "#e67e22",
    "PHASE_NOISE": "#9b59b6",
    "MULTIPLE":    "#c0392b",
    "INTERFERENCE":"#e74c3c",
    "IQ_IMBALANCE":"#f39c12",
    "LOW_SNR":     "#e67e22",
}

CATEGORY_ICONS = {
    "PASS":        "✅",
    "HIGH_EVM":    "❌",
    "HIGH_BER":    "⚠️",
    "PHASE_NOISE": "🔄",
    "MULTIPLE":    "🚨",
    "INTERFERENCE":"📡",
    "IQ_IMBALANCE":"⚖️",
    "LOW_SNR":     "📉",
}


def verdict_badge(category: str) -> None:
    color = CATEGORY_COLORS.get(category, "#95a5a6")
    icon  = CATEGORY_ICONS.get(category, "❓")
    st.markdown(
        f"""
        <div style="background:{color};padding:16px 24px;border-radius:12px;
                    display:inline-block;margin:8px 0;">
          <span style="color:white;font-size:1.6rem;font-weight:700;">
            {icon} {category}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_constellation(signal: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(signal.real, signal.imag, s=3, alpha=0.4, color="#3498db")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_amplitude_time(signal: np.ndarray) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(7, 3.5), sharex=True)
    t = np.arange(len(signal))
    axes[0].plot(t, signal.real, lw=0.7, color="#2980b9")
    axes[0].set_ylabel("I (real)")
    axes[0].grid(True, alpha=0.2)
    axes[1].plot(t, signal.imag, lw=0.7, color="#e74c3c")
    axes[1].set_ylabel("Q (imag)")
    axes[1].set_xlabel("Sample index")
    axes[1].grid(True, alpha=0.2)
    fig.suptitle("IQ Time Series", fontsize=10)
    fig.tight_layout()
    return fig


def plot_spectrum(signal: np.ndarray) -> plt.Figure:
    N = min(len(signal), 4096)
    freqs = np.fft.fftshift(np.fft.fftfreq(N))
    psd   = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(signal[:N], N))) + 1e-12)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(freqs, psd, lw=0.8, color="#8e44ad")
    ax.set_xlabel("Normalised frequency")
    ax.set_ylabel("Power (dB)")
    ax.set_title("Signal Spectrum")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def metrics_cards(feats: dict) -> None:
    cols = st.columns(4)
    items = [
        ("EVM est.", f"{feats.get('evm_est_percent', feats.get('evm_percent', 0)):.2f} %"),
        ("SNR est.", f"{feats.get('snr_est_db', feats.get('snr_db', 0)):.1f} dB"),
        ("Phase std", f"{feats.get('phase_std', 0):.4f} rad"),
        ("Kurtosis", f"{feats.get('kurtosis', 0):.3f}"),
    ]
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def render_report(report: dict, signal: np.ndarray) -> None:
    """Render the triage report with plots."""

    category = report.get("failure_category") or report.get("verdict", "?")

    st.subheader("Verdict")
    verdict_badge(category)

    st.subheader("Signal Metrics")
    feats = report.get("blind_features") or report.get("metrics") or {}
    metrics_cards(feats)

    # Plots row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.pyplot(plot_constellation(signal[:1024], "Constellation"))
    with col2:
        st.pyplot(plot_amplitude_time(signal[:512]))
    with col3:
        st.pyplot(plot_spectrum(signal))

    # Root cause + recommendations
    st.subheader("Root Cause")
    st.info(report.get("root_cause", ROOT_CAUSE_MAP.get(category, "")))

    recs = report.get("recommendations", RECOMMENDATIONS.get(category, []))
    if recs:
        st.subheader("Recommendations")
        for r in recs:
            st.markdown(f"- {r}")

    # ML details
    ml = report.get("ml_impairment_detection") or report.get("impairment_detection")
    if ml:
        with st.expander("ML Model Details"):
            col_a, col_b = st.columns(2)
            col_a.metric("ML Result", ml.get("result", ""))
            col_a.metric("PASS probability", f"{ml.get('pass_probability', 0):.1%}")
            col_b.metric("FAIL probability", f"{ml.get('fail_probability', 0):.1%}")

    # Rule-based (IQ upload only)
    rbd = report.get("rule_based_diagnosis")
    if rbd:
        with st.expander("Rule-Based Diagnosis"):
            st.write(f"**Issues detected:** {', '.join(rbd['issues_detected']) or 'None'}")
            st.write(f"**Detail:** {rbd['detail']}")

    # Full JSON
    with st.expander("Full Report JSON"):
        # Remove signal array from display
        display = {k: v for k, v in report.items() if k != "signal"}
        st.json(display)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/radio-tower.png", width=64)
    st.title("📡 Wireless Triage")
    st.markdown(
        "Upload real IQ data or simulate a signal to get an automated "
        "root-cause diagnosis of RF impairments."
    )
    st.divider()
    st.markdown("**Supported file formats**")
    st.markdown(
        "- `.csv`  — columns I, Q\n"
        "- `.npy`  — numpy complex array\n"
        "- `.bin`  — raw float32 interleaved\n"
        "- `.txt`  — space-separated I Q pairs"
    )
    st.divider()
    st.markdown("**Modulations**")
    st.markdown("BPSK · QPSK · 16-QAM")


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab_upload, tab_sim, tab_about = st.tabs(
    ["📂 Upload Real IQ Data", "🔬 Simulate Signal", "ℹ️ About"]
)


# ===========================
# TAB 1 — Real IQ Upload
# ===========================

with tab_upload:
    st.header("Upload Real IQ Data")
    st.markdown(
        "Upload IQ samples captured from real hardware (SDR, signal analyser, etc.) "
        "and get an automated diagnosis."
    )

    # Sample file download
    with st.expander("📥 Download a sample file to see the expected format"):
        s_col1, s_col2, s_col3 = st.columns(3)
        sample_mod = s_col1.selectbox("Modulation", ["BPSK", "QPSK", "16QAM"], index=1, key="smod")
        sample_snr = s_col2.slider("SNR (dB)", 5, 30, 15, key="ssnr")
        sample_n   = s_col3.selectbox("Symbols", [256, 512, 1024], index=1, key="sn")
        csv_bytes = generate_sample_iq_csv(sample_mod, sample_snr, sample_n)
        st.download_button(
            label=f"⬇️ Download sample_{sample_mod}_{sample_snr}dB.csv",
            data=csv_bytes,
            file_name=f"sample_iq_{sample_mod}_{sample_snr}dB.csv",
            mime="text/csv",
        )

    st.divider()

    # Upload form
    u_col1, u_col2 = st.columns([2, 1])
    uploaded_file = u_col1.file_uploader(
        "Choose your IQ file",
        type=["csv", "npy", "bin", "txt"],
        help="I/Q samples from your SDR, spectrum analyser, or test bench",
    )
    modulation = u_col2.selectbox(
        "Modulation scheme",
        ["BPSK", "QPSK", "16QAM"],
        index=1,
        help="What modulation was used to transmit this signal?",
    )

    if uploaded_file is not None:
        with st.spinner("Parsing file and running triage..."):
            try:
                raw_bytes = uploaded_file.read()
                signal = parse_iq_file(raw_bytes, uploaded_file.name)

                feats    = extract_features_blind(signal)
                diagnosis = diagnose_from_blind_features(feats, modulation)

                # ML path
                ml_features = {
                    "evm_percent":        feats["evm_est_percent"],
                    "amp_std":            feats["amplitude_std"],
                    "phase_std":          feats["phase_std"],
                    "kurtosis":           feats["kurtosis"],
                    "mean_amp_error":     feats["amplitude_std"],
                    "snr_db":             max(feats["snr_est_db"], 0),
                    "freq_offset":        abs(feats["freq_offset_est"]),
                    "interference_power": max(feats["kurtosis"] * 0.1, 0),
                    "modulation":         modulation,
                }
                imp_result = predict_impairment(ml_features)

                log_msg = (
                    f"[{'PASS' if imp_result['predicted_label'] == 1 else 'FAIL'}] {modulation} | "
                    f"SNR_est={feats['snr_est_db']:.1f}dB | "
                    f"FreqOffset_est={feats['freq_offset_est']:.4f} | "
                    f"EVM_est={feats['evm_est_percent']:.2f}% | "
                    f"PhaseStd={feats['phase_std']:.4f} | "
                    f"Kurtosis={feats['kurtosis']:.3f}"
                )
                cat_result = categorize_failure(log_msg)
                category   = cat_result["category"]

                report = {
                    "source":                  "real_iq_upload",
                    "filename":                uploaded_file.name,
                    "modulation":              modulation,
                    "blind_features":          feats,
                    "rule_based_diagnosis":    diagnosis,
                    "ml_impairment_detection": {
                        "result":           imp_result["label_name"],
                        "pass_probability": round(imp_result["pass_probability"], 4),
                        "fail_probability": round(imp_result["fail_probability"], 4),
                    },
                    "failure_category":  category,
                    "root_cause":        ROOT_CAUSE_MAP.get(category, ""),
                    "recommendations":   RECOMMENDATIONS.get(category, []),
                    "log_message":       log_msg,
                }

                st.success(f"Analysed {feats['n_samples']:,} IQ samples from **{uploaded_file.name}**")
                render_report(report, signal)

            except ValueError as e:
                st.error(f"Could not parse file: {e}")
            except Exception as e:
                st.error(f"Triage failed: {e}")
    else:
        st.info("⬆️ Upload an IQ file above to get started.")


# ===========================
# TAB 2 — Simulator
# ===========================

with tab_sim:
    st.header("Signal Simulator")
    st.markdown(
        "No hardware? No problem. Set channel conditions and see what the triage "
        "engine concludes."
    )

    with st.form("sim_form"):
        s1, s2, s3 = st.columns(3)
        mod     = s1.selectbox("Modulation", ["BPSK", "QPSK", "16QAM"], index=1)
        snr     = s2.slider("SNR (dB)", 5.0, 30.0, 12.0, step=0.5)
        f_off   = s3.slider("Frequency offset", 0.0, 0.05, 0.02, step=0.001, format="%.3f")

        s4, s5 = st.columns(2)
        interf  = s4.slider("Interference power", 0.0, 0.5, 0.1, step=0.01)
        n_sym   = s5.selectbox("Number of symbols", [128, 256, 512, 1024], index=1)

        submitted = st.form_submit_button("▶ Run Triage", use_container_width=True)

    if submitted:
        with st.spinner("Simulating signal and running triage..."):
            report = triage_case(
                modulation=mod,
                snr_db=snr,
                freq_offset=f_off,
                interference_power=interf,
                n_symbols=n_sym,
            )

            # Re-generate signal for plotting
            ref = MODULATORS[mod](n_sym)
            rx  = add_awgn(ref, snr)
            rx  = add_frequency_offset(rx, f_off)
            rx  = add_interference(rx, interf)

            # Adapt report keys for render_report
            report["blind_features"] = {
                "evm_est_percent": report["metrics"]["evm_percent"],
                "snr_est_db":      snr,
                "phase_std":       report["metrics"]["phase_std_rad"],
                "kurtosis":        report["metrics"]["kurtosis"],
                "amplitude_std":   0,
                "freq_offset_est": f_off,
            }
            render_report(report, rx)

            # Download simulated IQ
            df_dl = pd.DataFrame({"I": rx.real, "Q": rx.imag})
            st.download_button(
                "⬇️ Download simulated IQ as CSV",
                data=df_dl.to_csv(index=False).encode(),
                file_name=f"sim_{mod}_{snr}dB.csv",
                mime="text/csv",
            )


# ===========================
# TAB 3 — About
# ===========================

with tab_about:
    st.header("About this tool")
    st.markdown("""
    ### What it does
    This tool diagnoses RF signal impairments automatically using two machine learning models:

    | Model | Input | Output |
    |-------|-------|--------|
    | **Impairment Detector** (Random Forest) | Signal features: EVM, phase noise, kurtosis… | PASS / FAIL |
    | **Failure Categorizer** (TF-IDF + Logistic Regression) | Log message text | HIGH_EVM / HIGH_BER / PHASE_NOISE / MULTIPLE |

    ### Failure categories
    | Category | Meaning | Typical cause |
    |----------|---------|---------------|
    | ✅ PASS | All metrics within spec | — |
    | ❌ HIGH_EVM | Signal points too far from ideal | Noise, IQ imbalance, freq offset |
    | ⚠️ HIGH_BER | Too many bit errors | Low SNR, interference |
    | 🔄 PHASE_NOISE | Signal phase wobbling | Oscillator instability, Doppler |
    | 🚨 MULTIPLE | Several issues at once | Severe channel degradation |

    ### How to use with real hardware
    1. Capture IQ samples on your SDR / signal analyser
    2. Export as CSV (columns: I, Q)
    3. Upload in the **Upload Real IQ Data** tab
    4. Select the modulation scheme used
    5. Get your diagnosis instantly

    ### Stack
    Python · scikit-learn · NumPy · Pandas · Matplotlib · FastAPI · Streamlit · Docker · GitHub Actions
    """)
