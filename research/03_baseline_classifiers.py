"""
Hivemind — Stage 1, Step 3: Baseline Classifiers (CSP + LDA & PyRiemann)
========================================================================
Evaluates 3-class motor imagery (Rest, Left, Right) using strict
Leave-One-Run-Out cross-validation on PhysioNet EEGMMIDB (Runs 4, 8, 12).
Measures Accuracy, Macro-F1, Confusion Matrix, False Activation Rate, and
single-window inference latency (ms).
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "datasets" / "physionet"
OUT_DIR = ROOT_DIR / "results" / "research_03"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [4, 8, 12]
CLASSES = ["Rest (T0)", "Left (T1)", "Right (T2)"]


def load_subject_runs_epochs(subject: int, runs: list[int]):
    """Load, 8-30 Hz filter, and epoch each run separately (prevents train/test leakage)."""
    mne.set_log_level("WARNING")
    paths = eegbci.load_data(subject, runs, path=str(DATA_DIR), update_path=False)
    X_runs, y_runs = [], []

    for p in paths:
        raw = mne.io.read_raw_edf(p, preload=True)
        eegbci.standardize(raw)
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
        # Map event codes cleanly to 0=Rest, 1=Left, 2=Right
        code_to_cls = {event_id["T0"]: 0, event_id["T1"]: 1, event_id["T2"]: 2}
        y = np.array([code_to_cls[c] for c in epochs.events[:, -1]])
        X = epochs.get_data()  # shape: (n_trials, 64, n_times)
        X_runs.append(X)
        y_runs.append(y)

    return X_runs, y_runs


def make_csp_lda():
    return make_pipeline(
        CSP(n_components=6, reg="ledoit_wolf", log=True),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[1 / 3, 1 / 3, 1 / 3]),
    )


def make_riemann_lda():
    return make_pipeline(
        Covariances(estimator="oas"),
        TangentSpace(metric="riemann"),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[1 / 3, 1 / 3, 1 / 3]),
    )


def evaluate_subject(subject: int, model_fn):
    """Leave-One-Run-Out evaluation across Runs 4, 8, 12."""
    X_runs, y_runs = load_subject_runs_epochs(subject, RUNS)
    y_true_all, y_pred_all, latencies_ms = [], [], []

    for test_idx in range(len(RUNS)):
        X_train = np.concatenate([X_runs[i] for i in range(len(RUNS)) if i != test_idx], axis=0)
        y_train = np.concatenate([y_runs[i] for i in range(len(RUNS)) if i != test_idx], axis=0)
        X_test = X_runs[test_idx]
        y_test = y_runs[test_idx]

        model = model_fn()
        model.fit(X_train, y_train)

        # Measure single-window inference latency
        for sample_idx in range(len(X_test)):
            t0 = time.perf_counter()
            pred = model.predict(X_test[sample_idx : sample_idx + 1])[0]
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            y_pred_all.append(pred)
            y_true_all.append(y_test[sample_idx])

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    acc = accuracy_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all, average="macro")
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1, 2])

    # False Activation Rate: Rest (0) misclassified as Left (1) or Right (2)
    rest_mask = y_true_all == 0
    far = np.mean(y_pred_all[rest_mask] != 0) if np.any(rest_mask) else 0.0

    return {
        "acc": acc,
        "f1": f1,
        "cm": cm,
        "far": far,
        "latency_ms": float(np.mean(latencies_ms)),
    }


def plot_confusion_matrices(res_csp, res_rie) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    short_labels = ["Rest", "Left", "Right"]

    for ax, res, title in [
        (axes[0], res_csp, f"CSP + LDA\nAcc: {res_csp['acc']*100:.1f}% | F1: {res_csp['f1']:.2f} | FAR: {res_csp['far']*100:.1f}%"),
        (axes[1], res_rie, f"PyRiemann (TS + LDA)\nAcc: {res_rie['acc']*100:.1f}% | F1: {res_rie['f1']:.2f} | FAR: {res_rie['far']*100:.1f}%"),
    ]:
        cm = res["cm"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(short_labels)
        ax.set_yticklabels(short_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title, fontsize=10.5, fontweight="bold")

        thresh = cm.max() / 2.0
        for i in range(3):
            for j in range(3):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold",
                )

    fig.suptitle("Subject 001 (Leave-One-Run-Out, 3-Class): Confusion Matrices", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_multi_subject(subjects: list[int], csp_accs: list[float], rie_accs: list[float]) -> None:
    x = np.arange(len(subjects))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - width / 2, [a * 100 for a in csp_accs], width, label="CSP + LDA", color="#1d4ed8")
    ax.bar(x + width / 2, [a * 100 for a in rie_accs], width, label="PyRiemann (TS + LDA)", color="#b45309")
    ax.axhline(33.3, color="#71717a", linestyle="--", linewidth=1.2, label="Random Chance (33.3%)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s:03d}" for s in subjects])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("PhysioNet EEGMMIDB (Subjects 1–10): 3-Class Within-Subject Accuracy", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_subject_1_to_10_accuracy.png", dpi=150)
    plt.close(fig)


def main() -> None:
    print("=" * 72)
    print("STAGE 1 · STEP 3: BASELINE CLASSIFIERS (3-CLASS: REST / LEFT / RIGHT)")
    print("=" * 72)

    # 1. Detailed Subject 001 evaluation
    res_csp_s1 = evaluate_subject(1, make_csp_lda)
    res_rie_s1 = evaluate_subject(1, make_riemann_lda)
    plot_confusion_matrices(res_csp_s1, res_rie_s1)

    print("\nSubject 001 (Leave-One-Run-Out across Runs 4, 8, 12):")
    for name, r in [("CSP + LDA", res_csp_s1), ("PyRiemann (TS + LDA)", res_rie_s1)]:
        print(
            f"  {name:22s} | Acc: {r['acc']*100:5.1f}% | Macro-F1: {r['f1']:.3f} "
            f"| Rest FAR: {r['far']*100:5.1f}% | Latency: {r['latency_ms']:.2f} ms"
        )

    # 2. Multi-subject benchmark (Subjects 1-10)
    subjects = list(range(1, 11))
    csp_accs, rie_accs, csp_fars, rie_fars = [], [], [], []

    print("\nSubjects 1–10 Benchmark:")
    for s in subjects:
        r_c = evaluate_subject(s, make_csp_lda)
        r_r = evaluate_subject(s, make_riemann_lda)
        csp_accs.append(r_c["acc"])
        rie_accs.append(r_r["acc"])
        csp_fars.append(r_c["far"])
        rie_fars.append(r_r["far"])
        print(
            f"  S{s:03d} | CSP+LDA: {r_c['acc']*100:5.1f}% (FAR {r_c['far']*100:4.1f}%) "
            f"| PyRiemann: {r_r['acc']*100:5.1f}% (FAR {r_r['far']*100:4.1f}%)"
        )

    plot_multi_subject(subjects, csp_accs, rie_accs)

    print("\n" + "-" * 72)
    print(f"Mean Accuracy (S001–S010) -> CSP+LDA: {np.mean(csp_accs)*100:.1f}% | PyRiemann: {np.mean(rie_accs)*100:.1f}%")
    print(f"Mean Rest FAR (S001–S010) -> CSP+LDA: {np.mean(csp_fars)*100:.1f}% | PyRiemann: {np.mean(rie_fars)*100:.1f}%")
    print(f"Saved figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()

