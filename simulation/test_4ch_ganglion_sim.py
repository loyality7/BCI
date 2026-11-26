"""
BCI — 4-Channel Hardware-Realistic Test (C3, Cz, C4, Pz)
=========================================================
Tests whether a 4-channel EEG headset (e.g. OpenBCI Ganglion) is enough
for reliable 3-class calibration and Stage 2 causal hardware control.

1. Benchmarks PyRiemann (4ch) on PhysioNet S001-S010 (Runs 4+8 -> Run 12).
2. Runs full Stage 2 causal 160 Hz replay (threshold=0.60, debounce=3,
   artifact=150 uV, cooldown=2.0 s) on S004 & S007 using ONLY 4 channels.
"""

import sys
from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci
from scipy.signal import butter, sosfilt, sosfilt_zi
from sklearn.metrics import accuracy_score

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.stage1_classifier import build_pipeline, predict_window, save_model

DATA_DIR = ROOT_DIR / "datasets" / "physionet"
CH_4 = ["C3", "Cz", "C4", "Pz"]
SFREQ = 160.0
WINDOW_SAMPLES = int(3.0 * SFREQ)
STEP_SAMPLES = int(0.25 * SFREQ)


def load_epochs_4ch(subject: int, runs: list[int]):
    mne.set_log_level("WARNING")
    paths = eegbci.load_data(subject, runs, path=str(DATA_DIR), update_path=False)
    Xs, ys = [], []
    for p in paths:
        raw = mne.io.read_raw_edf(p, preload=True)
        eegbci.standardize(raw)
        raw.pick([ch for ch in CH_4 if ch in raw.ch_names])
        raw.filter(l_freq=8.0, h_freq=30.0, verbose=False)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        wanted = {"Rest": event_id["T0"], "Left": event_id["T1"], "Right": event_id["T2"]}
        epochs = mne.Epochs(raw, events, event_id=wanted, tmin=0.5, tmax=3.5, baseline=None, preload=True, verbose=False)
        code_to_cls = {event_id["T0"]: 0, event_id["T1"]: 1, event_id["T2"]: 2}
        y = np.array([code_to_cls[c] for c in epochs.events[:, -1]])
        Xs.append(epochs.get_data())
        ys.append(y)
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def eval_4ch_replay(subject: int, model):
    edf_path = eegbci.load_data(subject, [12], path=str(DATA_DIR), update_path=False)[0]
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    eegbci.standardize(raw)
    raw.pick([ch for ch in CH_4 if ch in raw.ch_names])
    raw_data = raw.get_data()

    truth = np.zeros(raw.n_times, dtype=int)
    for onset, dur, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        if desc in {"T0": 0, "T1": 1, "T2": 2}:
            truth[int(onset * SFREQ) : min(raw.n_times, int((onset + dur) * SFREQ))] = {"T0": 0, "T1": 1, "T2": 2}[desc]

    sos = butter(4, [8.0, 30.0], btype="bandpass", fs=SFREQ, output="sos")
    zi = np.repeat(sosfilt_zi(sos)[:, np.newaxis, :], len(CH_4), axis=1) * raw_data[:, 0][np.newaxis, :, np.newaxis]

    buffer = np.zeros((len(CH_4), WINDOW_SAMPLES), dtype=np.float64)
    samples_seen, led_on, streak_cls, streak_count, last_switch_t = 0, 0, 0, 0, -1e9
    switches, rest_switches, preds, trues = 0, 0, [], []

    for start in range(0, raw_data.shape[1] - STEP_SAMPLES + 1, STEP_SAMPLES):
        chunk = raw_data[:, start : start + STEP_SAMPLES]
        filt_chunk, zi = sosfilt(sos, chunk, axis=1, zi=zi)
        buffer = np.roll(buffer, -STEP_SAMPLES, axis=1)
        buffer[:, -STEP_SAMPLES:] = filt_chunk
        samples_seen += STEP_SAMPLES

        if samples_seen >= WINDOW_SAMPLES:
            cur_t = (start + STEP_SAMPLES) / SFREQ
            true_now = int(truth[start + STEP_SAMPLES - 1])
            peak_uv = float(np.max(np.abs(buffer)) * 1e6)

            if peak_uv > 150.0:
                pred = 0
                streak_cls, streak_count = 0, 0
            else:
                res = predict_window(model, buffer, active_threshold=0.60)
                pred = res["class_id"]
                if pred in (1, 2):
                    streak_count = streak_count + 1 if pred == streak_cls else 1
                    streak_cls = pred
                else:
                    streak_cls, streak_count = 0, 0

            if streak_count >= 3 and (cur_t - last_switch_t) >= 2.0:
                new_led = 1 if streak_cls == 1 else 0
                if new_led != led_on:
                    led_on = new_led
                    last_switch_t = cur_t
                    switches += 1
                    if true_now == 0:
                        rest_switches += 1

            preds.append(pred)
            trues.append(true_now)

    preds, trues = np.array(preds), np.array(trues)
    return float(np.mean(preds == trues)) * 100, switches, rest_switches


def main():
    print("=" * 74)
    print(f"4-CHANNEL HARDWARE TEST ({', '.join(CH_4)}) | Runs 4+8 -> Unseen Run 12")
    print("=" * 74)

    accs, fars = [], []
    models = {}
    for s in range(1, 11):
        X_cal, y_cal = load_epochs_4ch(s, [4, 8])
        X_te, y_te = load_epochs_4ch(s, [12])
        model = build_pipeline()
        model.fit(X_cal, y_cal)
        models[s] = model

        preds = [predict_window(model, X_te[i], active_threshold=0.55)["class_id"] for i in range(len(X_te))]
        preds = np.array(preds)
        acc = accuracy_score(y_te, preds)
        far = float(np.mean(preds[y_te == 0] != 0))
        accs.append(acc)
        fars.append(far)
        print(f"  S{s:03d} | 4ch Trial Acc: {acc*100:5.1f}% | Rest FAR: {far*100:4.1f}%")

    print("-" * 74)
    print(f"  Mean (S001–S010, 4ch) -> Acc: {np.mean(accs)*100:.1f}% | Rest FAR: {np.mean(fars)*100:.1f}%")

    print("\nStage 2 Causal 160 Hz Replay with ONLY 4 Channels:")
    for s in [4, 7]:
        save_model(models[s], ROOT_DIR / "models" / f"pyriemann_4ch_s{s:03d}.joblib", {"channels": CH_4})
        step_acc, switches, rest_sw = eval_4ch_replay(s, models[s])
        print(f"  S{s:03d} (4ch) -> Step Acc: {step_acc:5.1f}% | LED Switches: {switches:2d} (During Rest: {rest_sw})")


if __name__ == "__main__":
    main()

