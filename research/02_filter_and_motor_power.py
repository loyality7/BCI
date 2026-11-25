"""
BCI — Stage 1, Step 2: Filtering & Motor Cortex Power Visualization
========================================================================
Purpose:
    1. Apply an 8-30 Hz bandpass filter to Subject 1 (PhysioNet EEGMMIDB, runs 4, 8, 12).
    2. Visually compare Raw vs. Filtered EEG on Fp1 (blinks) and C3/C4 (motor cortex).
    3. Compute average Mu (8-12 Hz) + Beta (13-30 Hz) power across C3, Cz, C4
       for T0 (Rest), T1 (Left Fist Imagery), and T2 (Right Fist Imagery).
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "datasets" / "physionet"
OUT_DIR = ROOT_DIR / "results" / "research_02"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT = 1
RUNS = [4, 8, 12]


def load_combined_raw(subject: int, runs: list[int]) -> mne.io.BaseRaw:
    mne.set_log_level("WARNING")
    paths = eegbci.load_data(subject, runs, path=str(DATA_DIR), update_path=False)
    raws = []
    for p in paths:
        r = mne.io.read_raw_edf(p, preload=True)
        eegbci.standardize(r)
        raws.append(r)
    return mne.concatenate_raws(raws)


def plot_raw_vs_filtered(raw: mne.io.BaseRaw, filtered: mne.io.BaseRaw) -> None:
    """Show how 8-30 Hz filtering removes eye blinks (Fp1) and baseline drift."""
    channels = ["Fp1", "C3", "C4"]
    s_start, s_end = raw.time_as_index([0.0, 20.0])

    raw_data, times = raw.get_data(picks=channels, start=s_start, stop=s_end, return_times=True)
    filt_data, _ = filtered.get_data(picks=channels, start=s_start, stop=s_end, return_times=True)
    raw_uv = raw_data * 1e6
    filt_uv = filt_data * 1e6

    fig, axes = plt.subplots(len(channels), 2, figsize=(13, 7), sharex=True)
    for row, ch in enumerate(channels):
        axes[row, 0].plot(times, raw_uv[row], color="#52525b", linewidth=0.7)
        axes[row, 0].set_ylabel(f"{ch} (uV)", fontsize=9)
        axes[row, 0].grid(alpha=0.25)

        axes[row, 1].plot(times, filt_uv[row], color="#1d4ed8", linewidth=0.7)
        axes[row, 1].set_ylim(-60, 60)
        axes[row, 1].grid(alpha=0.25)

    axes[0, 0].set_title("BEFORE: Raw Unfiltered EEG (Drift + Blinks)", fontweight="bold")
    axes[0, 1].set_title("AFTER: 8-30 Hz Bandpass Filtered (Motor Band)", fontweight="bold")
    axes[-1, 0].set_xlabel("Time (seconds)")
    axes[-1, 1].set_xlabel("Time (seconds)")

    fig.suptitle(
        "Subject 001 (Run 4): Effect of 8-30 Hz Bandpass Filtering",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_raw_vs_filtered.png", dpi=150)
    plt.close(fig)


def plot_motor_power_by_class(filtered: mne.io.BaseRaw) -> None:
    """Compare 8-30 Hz power at C3, Cz, C4 across Rest (T0), Left (T1), Right (T2)."""
    events, event_id = mne.events_from_annotations(filtered, verbose=False)
    picks = mne.pick_channels(filtered.ch_names, include=["C3", "Cz", "C4"])

    # Use 0.5s to 3.5s after cue onset to capture steady motor imagery
    epochs = mne.Epochs(
        filtered,
        events,
        event_id={"Rest (T0)": event_id["T0"], "Left (T1)": event_id["T1"], "Right (T2)": event_id["T2"]},
        tmin=0.5,
        tmax=3.5,
        picks=picks,
        baseline=None,
        preload=True,
        verbose=False,
    )

    classes = ["Rest (T0)", "Left (T1)", "Right (T2)"]
    ch_names = epochs.ch_names  # order in epochs
    power_uv2 = {}
    for cls in classes:
        data_uv = epochs[cls].get_data() * 1e6  # (n_trials, 3, n_times)
        # Mean variance (power) across time and trials
        power_uv2[cls] = np.mean(np.var(data_uv, axis=2), axis=0)

    x = np.arange(len(ch_names))
    width = 0.25
    colors = {"Rest (T0)": "#71717a", "Left (T1)": "#1d4ed8", "Right (T2)": "#b91c1c"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cls in enumerate(classes):
        vals = power_uv2[cls]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=cls, color=colors[cls])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ch}\n({'Left Motor' if ch=='C3' else 'Right Motor' if ch=='C4' else 'Center'})" for ch in ch_names])
    ax.set_ylabel("8-30 Hz Band Power (uV^2)")
    ax.set_title(
        "Subject 001 (Runs 4, 8, 12): Motor Cortex Power by Imagined Task\n"
        "Notice how C3 vs C4 power shifts relative to Rest",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_motor_power_c3_cz_c4.png", dpi=150)
    plt.close(fig)

    print("\nMean 8-30 Hz Power (uV^2):")
    for cls in classes:
        row = " | ".join(f"{ch}: {val:.2f}" for ch, val in zip(ch_names, power_uv2[cls]))
        print(f"  {cls:11s} -> {row}")


def main() -> None:
    raw = load_combined_raw(SUBJECT, RUNS)
    filtered = raw.copy().filter(l_freq=8.0, h_freq=30.0, verbose=False)
    plot_raw_vs_filtered(raw, filtered)
    plot_motor_power_by_class(filtered)
    print(f"Saved visual figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()

