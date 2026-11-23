"""
Hivemind — Research Step 1: Inspect & Visualize Real PhysioNet EEGMMIDB Data
==============================================================================
Purpose:
    Before building any filters or AI models, inspect one real human subject
    (Subject 1, runs 4, 8, and 12: imagined left/right fist movement & rest)
    to understand:
      1. Signal dimensions, sampling rate, and channel layout (10-10 system).
      2. Event structure (T0 = Rest, T1 = Left fist, T2 = Right fist) and counts.
      3. What raw human EEG actually looks like in time and frequency domains.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "datasets" / "physionet"
OUT_DIR = ROOT_DIR / "results" / "research_01"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT = 1
RUNS = [4, 8, 12]  # Motor imagery: left vs right fist


def load_subject_runs(subject: int, runs: list[int]) -> list[mne.io.BaseRaw]:
    """Download (if needed) and load EDF+ runs for one PhysioNet EEGMMIDB subject."""
    mne.set_log_level("WARNING")
    file_paths = eegbci.load_data(subject, runs, path=str(DATA_DIR), update_path=False)
    raws = []
    for fpath in file_paths:
        raw = mne.io.read_raw_edf(fpath, preload=True)
        # Standardize channel names (removes trailing dots like 'Fc5.' -> 'FC5')
        eegbci.standardize(raw)
        raw.set_montage("standard_1005", on_missing="ignore")
        raws.append(raw)
    return raws


def summarize_runs(raws: list[mne.io.BaseRaw], runs: list[int]) -> None:
    """Print structured metadata and event counts for each run."""
    print("=" * 72)
    print(f"PHYSIONET EEGMMIDB — SUBJECT {SUBJECT:03d} | RUNS {runs}")
    print("=" * 72)

    total_counts = {"T0": 0, "T1": 0, "T2": 0}
    total_durations = {"T0": [], "T1": [], "T2": []}

    for run_id, raw in zip(runs, raws):
        sfreq = raw.info["sfreq"]
        n_ch = len(raw.ch_names)
        dur = raw.times[-1]
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        counts = {}
        for desc, dur_sec in zip(raw.annotations.description, raw.annotations.duration):
            counts[desc] = counts.get(desc, 0) + 1
            if desc in total_counts:
                total_counts[desc] += 1
                total_durations[desc].append(dur_sec)

        print(f"\nRun {run_id:02d}:")
        print(f"  Sampling rate : {sfreq:.1f} Hz")
        print(f"  Channels      : {n_ch} ({', '.join(raw.ch_names[:6])} ... {', '.join(raw.ch_names[-4:])})")
        print(f"  Duration      : {dur:.2f} s ({raw.n_times} samples)")
        print(f"  Annotation IDs: {event_id}")
        print(f"  Event counts  : {counts}")

    print("\n" + "-" * 72)
    print("COMBINED ACROSS RUNS 4, 8, 12:")
    for label, meaning in [("T0", "Rest"), ("T1", "Left Fist Imagery"), ("T2", "Right Fist Imagery")]:
        durs = total_durations[label]
        avg_d = np.mean(durs) if durs else 0.0
        print(f"  {label} ({meaning:18s}): {total_counts[label]:3d} events | avg duration = {avg_d:.2f} s")
    print("=" * 72)


def plot_montage(raw: mne.io.BaseRaw) -> None:
    """Plot 2D scalp sensor layout highlighting motor cortex channels (C3, Cz, C4)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    raw.plot_sensors(show_names=True, axes=ax, show=False)

    # Highlight C3, Cz, C4
    pos = raw.get_montage().get_positions()["ch_pos"]
    for ch_name, color in [("C3", "#d62728"), ("Cz", "#ff7f0e"), ("C4", "#1f77b4")]:
        if ch_name in pos:
            # plot_sensors uses 2D projected coordinates; find matching text object
            for txt in ax.texts:
                if txt.get_text() == ch_name:
                    txt.set_color(color)
                    txt.set_fontweight("bold")
                    txt.set_fontsize(11)

    ax.set_title(
        "64-Channel EEG Montage (10-10 System)\n"
        "Motor Cortex Focus: C3 (Left), Cz (Center), C4 (Right)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_sensor_montage.png", dpi=150)
    plt.close(fig)


def plot_raw_traces(raw: mne.io.BaseRaw) -> None:
    """Plot 16 seconds of raw EEG from key channels with T0/T1/T2 event markers."""
    channels = ["Fp1", "C3", "Cz", "C4", "O1", "O2"]
    available = [ch for ch in channels if ch in raw.ch_names]

    t_start, t_end = 0.0, 20.0
    s_start, s_end = raw.time_as_index([t_start, t_end])
    data, times = raw.get_data(picks=available, start=s_start, stop=s_end, return_times=True)
    data_uv = data * 1e6  # Convert Volts -> microvolts

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 8), sharex=True)
    color_map = {"T0": "#7f7f7f", "T1": "#1f77b4", "T2": "#d62728"}
    label_map = {"T0": "T0 (Rest)", "T1": "T1 (Left)", "T2": "T2 (Right)"}

    for idx, (ax, ch_name) in enumerate(zip(axes, available)):
        ax.plot(times, data_uv[idx], color="#222222", linewidth=0.7)
        ax.set_ylabel(f"{ch_name}\n(uV)", fontsize=9, rotation=0, labelpad=28, va="center")
        ax.grid(alpha=0.25)

        # Overlay annotations inside [t_start, t_end]
        for onset, dur, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
            if onset + dur >= t_start and onset <= t_end:
                c = color_map.get(desc, "#999999")
                ax.axvspan(max(onset, t_start), min(onset + dur, t_end), color=c, alpha=0.18)
                if idx == 0 and onset >= t_start:
                    ax.text(
                        onset + 0.15,
                        ax.get_ylim()[1] * 0.78,
                        label_map.get(desc, desc),
                        color=c,
                        fontsize=8,
                        fontweight="bold",
                    )

    axes[-1].set_xlabel("Time (seconds) — Run 4", fontsize=10)
    fig.suptitle(
        "Subject 001 (Run 4): Raw Unfiltered Human EEG Across Key Scalp Regions\n"
        "Shading = Task Windows | Fp1 (Frontal/Eye Blinks), C3/Cz/C4 (Motor), O1/O2 (Visual)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_raw_eeg_traces.png", dpi=150)
    plt.close(fig)


def plot_events_summary(raws: list[mne.io.BaseRaw], runs: list[int]) -> None:
    """Plot event timeline per run and overall class balance."""
    fig, (ax_tl, ax_bar) = plt.subplots(
        1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [2.2, 1]}
    )

    color_map = {"T0": "#7f7f7f", "T1": "#1f77b4", "T2": "#d62728"}
    counts = {"T0": 0, "T1": 0, "T2": 0}

    for row_idx, (run_id, raw) in enumerate(zip(runs, raws)):
        for onset, dur, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
            if desc in counts:
                counts[desc] += 1
                ax_tl.barh(
                    y=row_idx,
                    width=dur,
                    left=onset,
                    height=0.55,
                    color=color_map[desc],
                    edgecolor="white",
                    linewidth=0.5,
                )

    ax_tl.set_yticks(range(len(runs)))
    ax_tl.set_yticklabels([f"Run {r}" for r in runs])
    ax_tl.set_xlabel("Time in Recording (seconds)")
    ax_tl.set_title("Task Timeline Across Runs 4, 8, 12", fontweight="bold")
    ax_tl.grid(axis="x", alpha=0.25)

    # Legend
    for code, name in [("T0", "T0: Rest"), ("T1", "T1: Left Fist"), ("T2", "T2: Right Fist")]:
        ax_tl.barh(0, 0, color=color_map[code], label=name)
    ax_tl.legend(loc="upper right", fontsize=8)

    # Class balance bar chart
    labels = ["T0\n(Rest)", "T1\n(Left Fist)", "T2\n(Right Fist)"]
    vals = [counts["T0"], counts["T1"], counts["T2"]]
    bars = ax_bar.bar(labels, vals, color=[color_map["T0"], color_map["T1"], color_map["T2"]], width=0.55)
    for bar, val in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 1, str(val), ha="center", fontweight="bold")
    ax_bar.set_ylabel("Number of Trials")
    ax_bar.set_title("Class Distribution (Notice ~2:1 Rest Imbalance)", fontweight="bold")
    ax_bar.set_ylim(0, max(vals) * 1.18)
    ax_bar.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_event_timeline_and_counts.png", dpi=150)
    plt.close(fig)


def plot_psd(raw: mne.io.BaseRaw) -> None:
    """Plot Power Spectral Density (1-75 Hz) highlighting Mu (8-12 Hz) & Beta (13-30 Hz)."""
    psd = raw.compute_psd(fmin=1.0, fmax=75.0, verbose=False)
    freqs = psd.freqs
    data_db = 10 * np.log10(psd.get_data() * 1e12)  # V^2/Hz -> uV^2/Hz -> dB

    fig, ax = plt.subplots(figsize=(11, 5))
    # Plot all channels faintly, and highlight C3, Cz, C4
    for ch_idx in range(data_db.shape[0]):
        ax.plot(freqs, data_db[ch_idx], color="#cccccc", linewidth=0.6, alpha=0.6)

    for ch_name, color in [("C3", "#d62728"), ("Cz", "#ff7f0e"), ("C4", "#1f77b4")]:
        if ch_name in raw.ch_names:
            idx = raw.ch_names.index(ch_name)
            ax.plot(freqs, data_db[idx], color=color, linewidth=1.8, label=f"{ch_name} (Motor Cortex)")

    ax.axvspan(8, 12, color="#2ca02c", alpha=0.14, label="Mu Band (8-12 Hz)")
    ax.axvspan(13, 30, color="#9467bd", alpha=0.12, label="Beta Band (13-30 Hz)")
    ax.axvline(60, color="#000000", linestyle="--", linewidth=1.0, alpha=0.6, label="60 Hz Line Noise")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density (dB re 1 uV^2/Hz)")
    ax.set_title(
        "Subject 001 (Run 4): Frequency Spectrum (1-75 Hz)\n"
        "Why we use an 8-30 Hz Bandpass Filter: isolates Mu + Beta motor rhythms & removes drift/line noise",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_power_spectrum_psd.png", dpi=150)
    plt.close(fig)


def main() -> None:
    raws = load_subject_runs(SUBJECT, RUNS)
    summarize_runs(raws, RUNS)
    plot_montage(raws[0])
    plot_raw_traces(raws[0])
    plot_events_summary(raws, RUNS)
    plot_psd(raws[0])
    print(f"\nSaved 4 visual figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()

