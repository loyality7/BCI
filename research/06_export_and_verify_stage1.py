"""
BCI — Stage 1 Readiness Verification & Model Export
========================================================
Trains the winning PyRiemann 64-channel model on calibration runs (Runs 4 & 8),
exports it to models/, reloads it from disk, and verifies all Stage 1 gates
on the completely unseen Run 12.
"""

import sys
from pathlib import Path
import numpy as np
import mne
from mne.datasets import eegbci
from sklearn.metrics import accuracy_score

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.stage1_classifier import build_pipeline, load_model, predict_window, save_model

DATA_DIR = ROOT_DIR / "datasets" / "physionet"
MODELS_DIR = ROOT_DIR / "models"


def load_run(subject: int, run: int):
    mne.set_log_level("WARNING")
    path = eegbci.load_data(subject, [run], path=str(DATA_DIR), update_path=False)[0]
    raw = mne.io.read_raw_edf(path, preload=True)
    eegbci.standardize(raw)
    raw.filter(l_freq=8.0, h_freq=30.0, verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    wanted = {"Rest": event_id["T0"], "Left": event_id["T1"], "Right": event_id["T2"]}
    epochs = mne.Epochs(raw, events, event_id=wanted, tmin=0.5, tmax=3.5, baseline=None, preload=True, verbose=False)
    code_to_cls = {event_id["T0"]: 0, event_id["T1"]: 1, event_id["T2"]: 2}
    y = np.array([code_to_cls[c] for c in epochs.events[:, -1]])
    return epochs.get_data(), y


def verify_subject(subject: int, active_threshold: float = 0.50):
    X4, y4 = load_run(subject, 4)
    X8, y8 = load_run(subject, 8)
    X_cal = np.concatenate([X4, X8], axis=0)
    y_cal = np.concatenate([y4, y8], axis=0)

    X_test, y_test = load_run(subject, 12)

    model = build_pipeline()
    model.fit(X_cal, y_cal)

    out_path = save_model(
        model,
        MODELS_DIR / f"pyriemann_s{subject:03d}.joblib",
        metadata={"subject": subject, "sfreq": 160.0, "band": "8-30 Hz", "channels": 64},
    )

    loaded_model, meta = load_model(out_path)
    preds, latencies = [], []
    for i in range(len(X_test)):
        res = predict_window(loaded_model, X_test[i], active_threshold=active_threshold)
        preds.append(res["class_id"])
        latencies.append(res["latency_ms"])

    preds = np.array(preds)
    acc = accuracy_score(y_test, preds)
    rest_mask = y_test == 0
    far = float(np.mean(preds[rest_mask] != 0))
    mean_lat = float(np.mean(latencies))

    print(f"Subject {subject:03d} (Calibrated Runs 4+8 -> Unseen Run 12):")
    print(f"  Saved artifact : {out_path.name} ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Accuracy       : {acc*100:5.1f}%  (Gate >= 70%: {'PASS' if acc >= 0.70 else 'BELOW TARGET'})")
    print(f"  Rest FAR       : {far*100:5.1f}%  (Gate <= 10%: {'PASS' if far <= 0.10 else 'FAIL'})")
    print(f"  Latency        : {mean_lat:5.2f} ms (Gate < 50ms: {'PASS' if mean_lat < 50.0 else 'FAIL'})")


def main():
    print("=" * 72)
    print("STAGE 1 READINESS VERIFICATION & MODEL EXPORT")
    print("=" * 72)
    for subj in [1, 4, 7]:
        verify_subject(subj, active_threshold=0.50)
    print("=" * 72)


if __name__ == "__main__":
    main()

