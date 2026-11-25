"""
BCI — Stage 1, Step 4: Motor-Cortex Channel Selection & Two-Stage Gating
=============================================================================
Tests two targeted improvements on PhysioNet EEGMMIDB (Subjects 1-10, Runs 4, 8, 12):
  1. Restrict channels from 64 -> 21 central/motor electrodes (reduces blink/neck noise
     and shrinks Riemannian TangentSpace features from 2,080 -> 231).
  2. Two-Stage Gating:
       Stage A: Rest (0) vs. Active Imagery (1) with confidence threshold
       Stage B: Left (1) vs. Right (2) trained purely on active motor trials
"""

import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "datasets" / "physionet"
OUT_DIR = ROOT_DIR / "results" / "research_04"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [4, 8, 12]
MOTOR_CHANNELS = [
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
]


def load_subject_runs_motor(subject: int, runs: list[int], picks_motor: bool = True):
    mne.set_log_level("WARNING")
    paths = eegbci.load_data(subject, runs, path=str(DATA_DIR), update_path=False)
    X_runs, y_runs = [], []

    for p in paths:
        raw = mne.io.read_raw_edf(p, preload=True)
        eegbci.standardize(raw)
        if picks_motor:
            available = [ch for ch in MOTOR_CHANNELS if ch in raw.ch_names]
            raw.pick(available)
        raw.filter(l_freq=8.0, h_freq=30.0, verbose=False)

        events, event_id = mne.events_from_annotations(raw, verbose=False)
        wanted = {"Rest": event_id["T0"], "Left": event_id["T1"], "Right": event_id["T2"]}
        epochs = mne.Epochs(
            raw,
            events,
            event_id=wanted,
            tmin=0.5,
            tmax=3.5,
            baseline=None,
            preload=True,
            verbose=False,
        )
        code_to_cls = {event_id["T0"]: 0, event_id["T1"]: 1, event_id["T2"]: 2}
        y = np.array([code_to_cls[c] for c in epochs.events[:, -1]])
        X_runs.append(epochs.get_data())
        y_runs.append(y)

    return X_runs, y_runs


def make_riemann_pipeline(n_classes: int):
    priors = [1 / n_classes] * n_classes
    return make_pipeline(
        Covariances(estimator="oas"),
        TangentSpace(metric="riemann"),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=priors),
    )


def make_csp_pipeline(n_classes: int):
    priors = [1 / n_classes] * n_classes
    return make_pipeline(
        CSP(n_components=6, reg="ledoit_wolf", log=True),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=priors),
    )


def eval_flat(X_runs, y_runs, model_fn):
    y_true_all, y_pred_all, latencies = [], [], []
    for test_idx in range(len(X_runs)):
        X_train = np.concatenate([X_runs[i] for i in range(len(X_runs)) if i != test_idx], axis=0)
        y_train = np.concatenate([y_runs[i] for i in range(len(y_runs)) if i != test_idx], axis=0)
        X_test, y_test = X_runs[test_idx], y_runs[test_idx]

        model = model_fn(3)
        model.fit(X_train, y_train)
        for j in range(len(X_test)):
            t0 = time.perf_counter()
            pred = model.predict(X_test[j : j + 1])[0]
            latencies.append((time.perf_counter() - t0) * 1000.0)
            y_pred_all.append(pred)
            y_true_all.append(y_test[j])

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    rest_mask = y_true_all == 0
    return {
        "acc": accuracy_score(y_true_all, y_pred_all),
        "f1": f1_score(y_true_all, y_pred_all, average="macro"),
        "far": float(np.mean(y_pred_all[rest_mask] != 0)),
        "latency_ms": float(np.mean(latencies)),
    }


def eval_two_stage(X_runs, y_runs, model_fn, active_threshold: float = 0.55):
    y_true_all, y_pred_all, latencies = [], [], []
    for test_idx in range(len(X_runs)):
        X_train = np.concatenate([X_runs[i] for i in range(len(X_runs)) if i != test_idx], axis=0)
        y_train = np.concatenate([y_runs[i] for i in range(len(y_runs)) if i != test_idx], axis=0)
        X_test, y_test = X_runs[test_idx], y_runs[test_idx]

        # Stage A: Rest (0) vs Active (1)
        y_gate_train = (y_train != 0).astype(int)
        gate_model = model_fn(2)
        gate_model.fit(X_train, y_gate_train)

        # Stage B: Left (1) vs Right (2) on active trials only
        active_mask = y_train != 0
        dir_model = model_fn(2)
        dir_model.fit(X_train[active_mask], y_train[active_mask])

        for j in range(len(X_test)):
            t0 = time.perf_counter()
            x_win = X_test[j : j + 1]
            p_active = gate_model.predict_proba(x_win)[0, 1]
            if p_active < active_threshold:
                pred = 0
            else:
                pred = dir_model.predict(x_win)[0]
            latencies.append((time.perf_counter() - t0) * 1000.0)
            y_pred_all.append(pred)
            y_true_all.append(y_test[j])

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    rest_mask = y_true_all == 0
    return {
        "acc": accuracy_score(y_true_all, y_pred_all),
        "f1": f1_score(y_true_all, y_pred_all, average="macro"),
        "far": float(np.mean(y_pred_all[rest_mask] != 0)),
        "latency_ms": float(np.mean(latencies)),
    }


def main() -> None:
    subjects = list(range(1, 11))
    configs = {
        "Riemann 21ch (3-class)": [],
        "Riemann 21ch (2-stage)": [],
        "CSP+LDA 21ch (2-stage)": [],
    }

    print("=" * 78)
    print("STAGE 1 · STEP 4: 21 MOTOR CHANNELS + TWO-STAGE GATING (SUBJECTS 1–10)")
    print("=" * 78)

    for s in subjects:
        X_runs, y_runs = load_subject_runs_motor(s, RUNS, picks_motor=True)
        r_flat = eval_flat(X_runs, y_runs, make_riemann_pipeline)
        r_gate = eval_two_stage(X_runs, y_runs, make_riemann_pipeline, active_threshold=0.55)
        c_gate = eval_two_stage(X_runs, y_runs, make_csp_pipeline, active_threshold=0.55)

        configs["Riemann 21ch (3-class)"].append(r_flat)
        configs["Riemann 21ch (2-stage)"].append(r_gate)
        configs["CSP+LDA 21ch (2-stage)"].append(c_gate)

        print(
            f"S{s:03d} | Rie-3cls: {r_flat['acc']*100:5.1f}% (F1 {r_flat['f1']:.2f}, FAR {r_flat['far']*100:4.1f}%) "
            f"| Rie-2stg: {r_gate['acc']*100:5.1f}% (F1 {r_gate['f1']:.2f}, FAR {r_gate['far']*100:4.1f}%) "
            f"| CSP-2stg: {c_gate['acc']*100:5.1f}% (F1 {c_gate['f1']:.2f}, FAR {c_gate['far']*100:4.1f}%)"
        )

    print("\n" + "-" * 78)
    for name, res_list in configs.items():
        mean_acc = np.mean([r["acc"] for r in res_list]) * 100
        mean_f1 = np.mean([r["f1"] for r in res_list])
        mean_far = np.mean([r["far"] for r in res_list]) * 100
        mean_lat = np.mean([r["latency_ms"] for r in res_list])
        print(
            f"{name:24s} -> Mean Acc: {mean_acc:5.1f}% | Macro-F1: {mean_f1:.3f} "
            f"| Rest FAR: {mean_far:4.1f}% | Latency: {mean_lat:.2f} ms"
        )

    # Plot comparison
    x = np.arange(len(subjects))
    width = 0.26
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#1d4ed8", "#0f766e", "#b45309"]

    for idx, (name, res_list) in enumerate(configs.items()):
        accs = [r["acc"] * 100 for r in res_list]
        ax.bar(x + (idx - 1) * width, accs, width, label=name, color=colors[idx])

    ax.axhline(70.0, color="#18181b", linestyle=":", linewidth=1.2, label="Stage 1 Target (70%)")
    ax.axhline(33.3, color="#71717a", linestyle="--", linewidth=1.0, label="Random Chance (33.3%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s:03d}" for s in subjects])
    ax.set_ylabel("3-Class Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Effect of 21 Motor Channels + Two-Stage Gating (PhysioNet S001–S010)", fontweight="bold")
    ax.legend(ncol=2, fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_motor_ch_and_gating.png", dpi=150)
    plt.close(fig)
    print(f"Saved figure to: {OUT_DIR / '01_motor_ch_and_gating.png'}")


if __name__ == "__main__":
    main()

