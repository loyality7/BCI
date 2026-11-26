"""
BCI — Stage 2, Step 2: Real-Time Parameter & Subject Variations
===============================================================
Tests different confidence thresholds, debounce streaks, and subjects
on unseen Run 12 causal 160 Hz replay to find the cleanest hardware behavior.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import eegbci
from scipy.signal import butter, sosfilt, sosfilt_zi

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.stage1_classifier import load_model, predict_window

DATA_DIR = ROOT_DIR / "datasets" / "physionet"
OUT_DIR = ROOT_DIR / "results" / "stage2_01"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SFREQ = 160.0
WINDOW_SAMPLES = int(3.0 * SFREQ)
STEP_SAMPLES = int(0.25 * SFREQ)


def get_ground_truth(raw: mne.io.BaseRaw) -> np.ndarray:
    truth = np.zeros(raw.n_times, dtype=int)
    label_map = {"T0": 0, "T1": 1, "T2": 2}
    for onset, dur, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        if desc in label_map:
            s0 = int(onset * SFREQ)
            s1 = min(raw.n_times, int((onset + dur) * SFREQ))
            truth[s0:s1] = label_map[desc]
    return truth


def run_variation(subject: int, threshold: float, debounce: int):
    mne.set_log_level("WARNING")
    edf_path = eegbci.load_data(subject, [12], path=str(DATA_DIR), update_path=False)[0]
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    eegbci.standardize(raw)
    raw_data = raw.get_data()
    truth = get_ground_truth(raw)

    model, _ = load_model(ROOT_DIR / "models" / f"pyriemann_s{subject:03d}.joblib")

    sos = butter(4, [8.0, 30.0], btype="bandpass", fs=SFREQ, output="sos")
    zi = np.repeat(sosfilt_zi(sos)[:, np.newaxis, :], raw_data.shape[0], axis=1)
    zi = zi * raw_data[:, 0][np.newaxis, :, np.newaxis]

    buffer = np.zeros((raw_data.shape[0], WINDOW_SAMPLES), dtype=np.float64)
    samples_seen = 0
    led_on = 0
    streak_cls = 0
    streak_count = 0

    preds, trues = [], []
    led_switches = 0
    wrong_rest_switches = 0

    for start in range(0, raw_data.shape[1] - STEP_SAMPLES + 1, STEP_SAMPLES):
        chunk = raw_data[:, start : start + STEP_SAMPLES]
        filt_chunk, zi = sosfilt(sos, chunk, axis=1, zi=zi)
        buffer = np.roll(buffer, -STEP_SAMPLES, axis=1)
        buffer[:, -STEP_SAMPLES:] = filt_chunk
        samples_seen += STEP_SAMPLES

        if samples_seen >= WINDOW_SAMPLES:
            res = predict_window(model, buffer, active_threshold=threshold)
            pred = res["class_id"]
            true_now = int(truth[start + STEP_SAMPLES - 1])
            preds.append(pred)
            trues.append(true_now)

            if pred in (1, 2):
                if pred == streak_cls:
                    streak_count += 1
                else:
                    streak_cls = pred
                    streak_count = 1
            else:
                streak_cls = 0
                streak_count = 0

            if streak_count >= debounce:
                new_led = 1 if streak_cls == 1 else 0
                if new_led != led_on:
                    led_on = new_led
                    led_switches += 1
                    if true_now == 0:
                        wrong_rest_switches += 1

    preds = np.array(preds)
    trues = np.array(trues)
    acc = float(np.mean(preds == trues)) * 100
    rest_far = float(np.mean(preds[trues == 0] != 0)) * 100
    return {
        "subject": subject,
        "threshold": threshold,
        "debounce": debounce,
        "acc": acc,
        "rest_far": rest_far,
        "switches": led_switches,
        "rest_switches": wrong_rest_switches,
    }


def main():
    variations = [
        (4, 0.50, 1),
        (4, 0.50, 2),
        (4, 0.50, 3),
        (4, 0.60, 2),
        (4, 0.60, 3),
        (4, 0.70, 2),
        (7, 0.50, 2),
        (7, 0.60, 3),
    ]

    print("=" * 78)
    print("STAGE 2 VARIATIONS: THRESHOLD, DEBOUNCE & SUBJECT COMPARISON (RUN 12)")
    print("=" * 78)

    results = []
    for subj, thr, deb in variations:
        r = run_variation(subj, thr, deb)
        results.append(r)
        print(
            f"  S{r['subject']:03d} | Thr={r['threshold']:.2f} | Deb={r['debounce']} "
            f"-> Step Acc: {r['acc']:5.1f}% | Rest FAR: {r['rest_far']:4.1f}% "
            f"| LED Switches: {r['switches']:2d} (During Rest: {r['rest_switches']})"
        )

    # Plot Subject 004 variations
    s4 = [r for r in results if r["subject"] == 4]
    labels = [f"T={r['threshold']:.2f}\nD={r['debounce']}" for r in s4]
    x = np.arange(len(s4))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar(x - 0.18, [r["acc"] for r in s4], 0.36, label="Step Accuracy (%)", color="#1d4ed8")
    ax1.bar(x + 0.18, [r["rest_far"] for r in s4], 0.36, label="Rest FAR (%)", color="#b91c1c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 100)
    ax1.set_title("Subject 004: Accuracy vs. Rest FAR", fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.25)

    ax2.bar(x - 0.18, [r["switches"] for r in s4], 0.36, label="Total LED Switches", color="#0f766e")
    ax2.bar(x + 0.18, [r["rest_switches"] for r in s4], 0.36, label="Switches During Rest", color="#b45309")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title("Subject 004: Hardware Switching Stability", fontweight="bold")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out_png = OUT_DIR / "02_stage2_variations.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved variations plot: {out_png}")


if __name__ == "__main__":
    main()

