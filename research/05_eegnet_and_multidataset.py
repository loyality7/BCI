"""
BCI — Stage 1, Step 5: EEGNet & Multi-Dataset Benchmark
============================================================
1. Benchmarks PyRiemann (TS + LDA) across 4 additional cached MOABB datasets:
     - Kaya2018 (3-class: left_hand, right_hand, rest)
     - BNCI2014_001 (4-class: left, right, feet, tongue)
     - BNCI2014_004 (2-class, 3 channels: C3, Cz, C4)
     - Zhou2016 (3-class: left, right, feet)
2. Compares PyRiemann vs. EEGNet (Braindecode/PyTorch) on PhysioNet (S001-S010):
     - Within-subject Leave-One-Run-Out
     - Cross-subject (train on S001-S008, test on unseen S009-S010)
"""

import os
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "datasets"
OUT_DIR = ROOT_DIR / "results" / "research_05"
OUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["MNE_DATA"] = str(DATA_DIR)
os.environ["MNE_DATASETS_EEGBCI_PATH"] = str(DATA_DIR / "physionet")

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from braindecode.models import EEGNet
from moabb.datasets import BNCI2014_001, BNCI2014_004, Kaya2018, Zhou2016
from moabb.paradigms import MotorImagery
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

mne.set_log_level("WARNING")
RUNS = [4, 8, 12]


def make_riemann(n_classes: int):
    return make_pipeline(
        Covariances(estimator="oas"),
        TangentSpace(metric="riemann"),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[1 / n_classes] * n_classes),
    )


def benchmark_moabb_datasets():
    """Evaluate PyRiemann (TS + LDA) across 4 additional cached datasets."""
    specs = [
        ("Kaya2018 (3-cls)", Kaya2018(), [1, 2, 3], 3, ["left_hand", "right_hand", "rest"]),
        ("BNCI2014_001 (4-cls)", BNCI2014_001(), list(range(1, 10)), 4, None),
        ("BNCI2014_004 (2-cls, 3ch)", BNCI2014_004(), list(range(1, 10)), 2, None),
    ]

    summary = []
    print("\n" + "=" * 72)
    print("PART A: MULTI-DATASET CHECK (PyRiemann TS + LDA)")
    print("=" * 72)

    for label, ds, subjs, n_cls, events in specs:
        try:
            paradigm = MotorImagery(n_classes=n_cls, events=events, fmin=8, fmax=30)
            subj_accs = []
            for s in subjs:
                X, y_str, _ = paradigm.get_data(dataset=ds, subjects=[s])
                _, y = np.unique(y_str, return_inverse=True)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                fold_accs = []
                for tr, te in skf.split(X, y):
                    clf = make_riemann(len(np.unique(y)))
                    clf.fit(X[tr], y[tr])
                    fold_accs.append(accuracy_score(y[te], clf.predict(X[te])))
                subj_accs.append(np.mean(fold_accs))

            mean_acc = float(np.mean(subj_accs)) * 100
            chance = 100.0 / n_cls
            summary.append((label, mean_acc, chance, len(subjs)))
            print(f"  {label:28s} | Subjects: {len(subjs):2d} | Mean Acc: {mean_acc:5.1f}% (Chance: {chance:.1f}%)")
        except Exception as exc:
            print(f"  {label:28s} | Skipped ({exc})")

    return summary


def load_physionet_subject_runs(subject: int):
    paths = eegbci.load_data(subject, RUNS, path=str(DATA_DIR / "physionet"), update_path=False)
    X_runs, y_runs = [], []
    for p in paths:
        raw = mne.io.read_raw_edf(p, preload=True)
        eegbci.standardize(raw)
        raw.filter(l_freq=8.0, h_freq=30.0, verbose=False)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        wanted = {"Rest": event_id["T0"], "Left": event_id["T1"], "Right": event_id["T2"]}
        epochs = mne.Epochs(raw, events, event_id=wanted, tmin=0.5, tmax=3.5, baseline=None, preload=True, verbose=False)
        code_to_cls = {event_id["T0"]: 0, event_id["T1"]: 1, event_id["T2"]: 2}
        y = np.array([code_to_cls[c] for c in epochs.events[:, -1]])
        X_runs.append(epochs.get_data().astype(np.float32))
        y_runs.append(y)
    return X_runs, y_runs


def train_and_eval_eegnet(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, epochs: int = 45):
    """Train compact EEGNetv4 on standardized EEG windows and return predictions + latency."""
    torch.manual_seed(42)
    # Channel-wise z-score normalization using training stats
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    X_tr = (X_train - mean) / std
    X_te = (X_test - mean) / std

    n_chans, n_times = X_tr.shape[1], X_tr.shape[2]
    model = EEGNet(n_chans=n_chans, n_outputs=3, n_times=n_times, drop_prob=0.35)

    # Inverse-frequency weights for Rest (2x) vs Left/Right
    counts = np.bincount(y_train, minlength=3).astype(np.float32)
    weights = torch.tensor(counts.sum() / (3.0 * np.maximum(counts, 1.0)), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_train).long()), batch_size=16, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, latencies = [], []
    with torch.no_grad():
        for i in range(len(X_te)):
            t0 = time.perf_counter()
            out = model(torch.from_numpy(X_te[i : i + 1]))
            latencies.append((time.perf_counter() - t0) * 1000.0)
            preds.append(int(out.argmax(dim=1).item()))

    return np.array(preds), float(np.mean(latencies))


def benchmark_eegnet_physionet():
    print("\n" + "=" * 72)
    print("PART B: EEGNet vs. PyRiemann ON PHYSIONET (S001-S010)")
    print("=" * 72)

    subjects = list(range(1, 11))
    all_X, all_y = {}, {}
    rie_accs, net_accs, rie_fars, net_fars, net_lats = [], [], [], [], []

    for s in subjects:
        X_runs, y_runs = load_physionet_subject_runs(s)
        all_X[s] = np.concatenate(X_runs, axis=0)
        all_y[s] = np.concatenate(y_runs, axis=0)

        y_true_all, rie_pred_all, net_pred_all = [], [], []
        for test_idx in range(len(RUNS)):
            X_tr = np.concatenate([X_runs[i] for i in range(len(RUNS)) if i != test_idx], axis=0)
            y_tr = np.concatenate([y_runs[i] for i in range(len(RUNS)) if i != test_idx], axis=0)
            X_te, y_te = X_runs[test_idx], y_runs[test_idx]

            # PyRiemann
            rie = make_riemann(3)
            rie.fit(X_tr.astype(np.float64), y_tr)
            rie_pred_all.extend(rie.predict(X_te.astype(np.float64)))

            # EEGNet
            net_preds, lat = train_and_eval_eegnet(X_tr, y_tr, X_te, epochs=45)
            net_pred_all.extend(net_preds)
            net_lats.append(lat)
            y_true_all.extend(y_te)

        y_true_all = np.array(y_true_all)
        rie_pred_all = np.array(rie_pred_all)
        net_pred_all = np.array(net_pred_all)
        rest_mask = y_true_all == 0

        r_acc = accuracy_score(y_true_all, rie_pred_all)
        n_acc = accuracy_score(y_true_all, net_pred_all)
        r_far = float(np.mean(rie_pred_all[rest_mask] != 0))
        n_far = float(np.mean(net_pred_all[rest_mask] != 0))

        rie_accs.append(r_acc)
        net_accs.append(n_acc)
        rie_fars.append(r_far)
        net_fars.append(n_far)
        print(
            f"  S{s:03d} | PyRiemann: {r_acc*100:5.1f}% (FAR {r_far*100:4.1f}%) "
            f"| EEGNet: {n_acc*100:5.1f}% (FAR {n_far*100:4.1f}%)"
        )

    print("-" * 72)
    print(
        f"  Within-Subject Mean -> PyRiemann: {np.mean(rie_accs)*100:.1f}% (FAR {np.mean(rie_fars)*100:.1f}%) "
        f"| EEGNet: {np.mean(net_accs)*100:.1f}% (FAR {np.mean(net_fars)*100:.1f}%, Latency {np.mean(net_lats):.2f} ms)"
    )

    # Cross-subject test: Train on S001-S008, test on unseen S009-S010
    X_cross_tr = np.concatenate([all_X[s] for s in range(1, 9)], axis=0)
    y_cross_tr = np.concatenate([all_y[s] for s in range(1, 9)], axis=0)
    X_cross_te = np.concatenate([all_X[9], all_X[10]], axis=0)
    y_cross_te = np.concatenate([all_y[9], all_y[10]], axis=0)

    rie_cross = make_riemann(3)
    rie_cross.fit(X_cross_tr.astype(np.float64), y_cross_tr)
    rie_cross_acc = accuracy_score(y_cross_te, rie_cross.predict(X_cross_te.astype(np.float64))) * 100

    net_cross_preds, _ = train_and_eval_eegnet(X_cross_tr, y_cross_tr, X_cross_te, epochs=45)
    net_cross_acc = accuracy_score(y_cross_te, net_cross_preds) * 100

    print(f"  Cross-Subject (Train S1-S8 -> Test S9-S10) -> PyRiemann: {rie_cross_acc:.1f}% | EEGNet: {net_cross_acc:.1f}%")

    return subjects, rie_accs, net_accs


def plot_results(moabb_summary, subjects, rie_accs, net_accs):
    if moabb_summary:
        labels = [m[0] for m in moabb_summary]
        accs = [m[1] for m in moabb_summary]
        chances = [m[2] for m in moabb_summary]
        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(9, 4.5))
        bars = ax.bar(x, accs, width=0.5, color="#1d4ed8", label="PyRiemann (TS + LDA)")
        ax.scatter(x, chances, color="#b91c1c", marker="_", s=400, linewidths=2.5, zorder=5, label="Random Chance")
        for b, v in zip(bars, accs):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Multi-Dataset Validation: PyRiemann Across 4 Additional Public Datasets", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "01_multidataset_pyriemann.png", dpi=150)
        plt.close(fig)

    x = np.arange(len(subjects))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - width / 2, [a * 100 for a in rie_accs], width, label="PyRiemann (TS + LDA)", color="#1d4ed8")
    ax.bar(x + width / 2, [a * 100 for a in net_accs], width, label="EEGNetv4 (PyTorch)", color="#0f766e")
    ax.axhline(70.0, color="#18181b", linestyle=":", linewidth=1.2, label="Stage 1 Target (70%)")
    ax.axhline(33.3, color="#71717a", linestyle="--", linewidth=1.0, label="Random Chance (33.3%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s:03d}" for s in subjects])
    ax.set_ylabel("3-Class Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("PyRiemann vs. EEGNetv4 (PhysioNet S001–S010, Within-Subject)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_eegnet_vs_pyriemann.png", dpi=150)
    plt.close(fig)


def main():
    moabb_summary = benchmark_moabb_datasets()
    subjects, rie_accs, net_accs = benchmark_eegnet_physionet()
    plot_results(moabb_summary, subjects, rie_accs, net_accs)
    print(f"\nSaved figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()

