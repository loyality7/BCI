"""
BCI — Stage 2, Step 1: 160 Hz Real-Time Causal Replay & Hardware Controller
==========================================================================
Simulates a live EEG headset by streaming Subject 004's unseen Run 12 (.edf)
chunk-by-chunk at 160 Hz, applying a stateful causal 8-30 Hz bandpass filter,
running pyriemann_s004.joblib on a 3-second sliding buffer, debouncing
predictions, and optionally driving an Arduino/ESP32 LED over USB Serial.

Usage:
    .venv/Scripts/python.exe simulation/realtime_replay.py --speed 10
    .venv/Scripts/python.exe simulation/realtime_replay.py --speed 1.0 --port COM3
"""

import argparse
import sys
import time
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
MODEL_PATH = ROOT_DIR / "models" / "pyriemann_s004.joblib"
OUT_DIR = ROOT_DIR / "results" / "stage2_01"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SFREQ = 160.0
WINDOW_SEC = 3.0
STEP_SEC = 0.25
WINDOW_SAMPLES = int(WINDOW_SEC * SFREQ)  # 480 samples
STEP_SAMPLES = int(STEP_SEC * SFREQ)      # 40 samples (4 predictions/sec)


def get_ground_truth_timeline(raw: mne.io.BaseRaw) -> np.ndarray:
    """Map every sample index in raw to its true annotation (0=Rest, 1=Left, 2=Right)."""
    truth = np.zeros(raw.n_times, dtype=int)
    label_map = {"T0": 0, "T1": 1, "T2": 2}
    for onset, dur, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        if desc in label_map:
            s0 = int(onset * SFREQ)
            s1 = min(raw.n_times, int((onset + dur) * SFREQ))
            truth[s0:s1] = label_map[desc]
    return truth


def plot_replay_timeline(times, true_cls, pred_cls, led_state, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 6.5), sharex=True)

    axes[0].step(times, true_cls, where="post", color="#18181b", linewidth=1.4)
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_yticklabels(["Rest", "Left", "Right"])
    axes[0].set_title("Actual Task Cue (Unseen Run 12)", fontweight="bold", fontsize=10.5)
    axes[0].grid(alpha=0.25)

    axes[1].step(times, pred_cls, where="post", color="#1d4ed8", linewidth=1.3)
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["Rest", "Left", "Right"])
    axes[1].set_title("Live Causal Model Prediction (Every 0.25 s)", fontweight="bold", fontsize=10.5)
    axes[1].grid(alpha=0.25)

    axes[2].step(times, led_state, where="post", color="#b45309", linewidth=1.8)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["OFF", "ON"])
    axes[2].set_title("Debounced Hardware LED State (2-Hit Agreement)", fontweight="bold", fontsize=10.5)
    axes[2].set_xlabel("Recording Time (seconds)")
    axes[2].grid(alpha=0.25)

    fig.suptitle(
        "Stage 2 Real-Time Simulation (Subject 004, Unseen Run 12 at 160 Hz)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="160 Hz real-time causal EEG replay")
    parser.add_argument("--subject", type=int, default=4)
    parser.add_argument("--run", type=int, default=12)
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier (1.0 = real-time)")
    parser.add_argument("--threshold", type=float, default=0.60, help="Confidence gate for Left/Right")
    parser.add_argument("--debounce", type=int, default=3, help="Consecutive hits required to switch LED")
    parser.add_argument("--artifact-uv", type=float, default=150.0, help="Freeze inference if 8-30 Hz spike exceeds uV")
    parser.add_argument("--cooldown", type=float, default=2.0, help="Seconds to lock LED after a switch")
    parser.add_argument("--port", type=str, default=None, help="Optional COM port for Arduino/ESP32 (e.g. COM5)")
    args = parser.parse_args()

    mne.set_log_level("WARNING")
    edf_path = eegbci.load_data(args.subject, [args.run], path=str(DATA_DIR), update_path=False)[0]
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    eegbci.standardize(raw)
    raw_data = raw.get_data()  # (64, n_times), UNFILTERED
    truth_per_sample = get_ground_truth_timeline(raw)

    model_file = ROOT_DIR / "models" / f"pyriemann_s{args.subject:03d}.joblib"
    model, _ = load_model(model_file)

    # Causal 4th-order Butterworth 8-30 Hz bandpass filter with persistent state across chunks
    sos = butter(4, [8.0, 30.0], btype="bandpass", fs=SFREQ, output="sos")
    zi_base = sosfilt_zi(sos)  # (n_sections, 2)
    zi = np.repeat(zi_base[:, np.newaxis, :], raw_data.shape[0], axis=1)
    # Initialize filter state with first sample to avoid startup transient
    zi = zi * raw_data[:, 0][np.newaxis, :, np.newaxis]
    # Transpose zi to (n_sections, 64, 2) -> scipy expects (n_sections, ..., 2) with axis=-1
    zi = np.transpose(zi, (0, 1, 2))

    ser = None
    if args.port:
        import serial
        ser = serial.Serial(args.port, 115200, timeout=0.1)
        time.sleep(3.0)  # allow Arduino auto-reset + 3-blink boot check
        print(f"Connected to hardware on {args.port} (115200 baud)")

    buffer = np.zeros((raw_data.shape[0], WINDOW_SAMPLES), dtype=np.float64)
    samples_seen = 0
    led_on = 0
    streak_cls = 0
    streak_count = 0
    last_switch_t = -1e9
    artifact_blocked = 0

    log_times, log_true, log_pred, log_led, step_latencies = [], [], [], [], []

    print("=" * 72)
    print(f"STAGE 2 REAL-TIME REPLAY | Subject {args.subject:03d} Run {args.run:02d} | Speed {args.speed}x")
    print("=" * 72)

    for start in range(0, raw_data.shape[1] - STEP_SAMPLES + 1, STEP_SAMPLES):
        t_step_start = time.perf_counter()
        chunk = raw_data[:, start : start + STEP_SAMPLES]

        # Causal filtering on this 0.25 s chunk only
        filt_chunk, zi = sosfilt(sos, chunk, axis=1, zi=zi)

        # Slide ring buffer
        buffer = np.roll(buffer, -STEP_SAMPLES, axis=1)
        buffer[:, -STEP_SAMPLES:] = filt_chunk
        samples_seen += STEP_SAMPLES

        if samples_seen >= WINDOW_SAMPLES:
            cur_t = (start + STEP_SAMPLES) / SFREQ
            true_now = int(truth_per_sample[start + STEP_SAMPLES - 1])
            peak_uv = float(np.max(np.abs(buffer)) * 1e6)

            # Edge-case guard: freeze inference if filtered buffer contains a sharp artifact spike
            if peak_uv > args.artifact_uv:
                pred = 0
                label_str = "Artif"
                streak_cls = 0
                streak_count = 0
                artifact_blocked += 1
            else:
                res = predict_window(model, buffer, active_threshold=args.threshold)
                pred = res["class_id"]
                label_str = res["label"]

                # Debounce: require N consecutive Left (1) or Right (2) predictions
                if pred in (1, 2):
                    if pred == streak_cls:
                        streak_count += 1
                    else:
                        streak_cls = pred
                        streak_count = 1
                else:
                    streak_cls = 0
                    streak_count = 0

            prev_led = led_on
            in_cooldown = (cur_t - last_switch_t) < args.cooldown
            if streak_count >= args.debounce and not in_cooldown:
                new_led = 1 if streak_cls == 1 else 0
                if new_led != led_on:
                    led_on = new_led
                    last_switch_t = cur_t
                    if ser is not None:
                        ser.write(b"1\n" if led_on else b"0\n")

            proc_ms = (time.perf_counter() - t_step_start) * 1000.0
            step_latencies.append(proc_ms)

            log_times.append(cur_t)
            log_true.append(true_now)
            log_pred.append(pred)
            log_led.append(led_on)

            if led_on != prev_led:
                print(
                    f"  t={cur_t:6.2f}s | Pred={label_str:5s} (True={['Rest','Left','Right'][true_now]:5s}) "
                    f"-> LED {'ON ' if led_on else 'OFF'} | Latency={proc_ms:.2f} ms"
                )

        # Pace replay to target speed
        elapsed = time.perf_counter() - t_step_start
        target_dt = STEP_SEC / args.speed
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)

    if ser is not None:
        ser.close()

    out_png = OUT_DIR / "01_realtime_replay.png"
    plot_replay_timeline(log_times, log_true, log_pred, log_led, out_png)

    acc = np.mean(np.array(log_pred) == np.array(log_true)) * 100
    print("-" * 72)
    print(
        f"Step Accuracy (every 0.25 s): {acc:.1f}% | Mean End-to-End Latency: {np.mean(step_latencies):.2f} ms "
        f"| Artifact Windows Blocked: {artifact_blocked}"
    )
    print(f"Saved timeline plot: {out_png}")


if __name__ == "__main__":
    main()

