"""
BCI — 4-Channel Filter-Bank vs. 8-Channel (Cyton-Equivalent) Test
=================================================================
Checks whether:
  A) Filter-Bank (Mu 8-12 Hz + Beta 13-30 Hz) rescues 4-channel accuracy, OR
  B) An 8-channel motor+parietal layout (FC3, FC4, C3, Cz, C4, CP3, CP4, Pz)
     is the sweet spot for reliable hardware control.
"""

import sys
from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion, make_pipeline

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
DATA_DIR = ROOT_DIR / "datasets" / "physionet"

CH_4 = ["C3", "Cz", "C4", "Pz"]
CH_8 = ["FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4", "Pz"]


class BandSelect(BaseEstimator, TransformerMixin):
    def __init__(self, idx: int):
        self.idx = idx

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.idx]


def load_band_epochs(subject: int, runs: list[int], channels: list[str], bands=((8.0, 12.0), (13.0, 30.0))):
    mne.set_log_level("WARNING")
    paths = eegbci.load_data(subject, runs, path=str(DATA_DIR), update_path=False)
    band_Xs = [[] for _ in bands]
    ys = []

    for p in paths:
        raw = mne.io.read_raw_edf(p, preload=True)
        eegbci.standardize(raw)
        raw.pick([ch for ch in channels if ch in raw.ch_names])
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        wanted = {"Rest": event_id["T0"], "Left": event_id["T1"], "Right": event_id["T2"]}

        for b_idx, (fmin, fmax) in enumerate(bands):
            r_filt = raw.copy().filter(l_freq=fmin, h_freq=fmax, verbose=False)
            ep = mne.Epochs(r_filt, events, event_id=wanted, tmin=0.5, tmax=3.5, baseline=None, preload=True, verbose=False)
            band_Xs[b_idx].append(ep.get_data())
            if b_idx == 0:
                code_to_cls = {event_id["T0"]: 0, event_id["T1"]: 1, event_id["T2"]: 2}
                ys.append(np.array([code_to_cls[c] for c in ep.events[:, -1]]))

    return [np.concatenate(bx, axis=0) for bx in band_Xs], np.concatenate(ys, axis=0)


def make_fb_riemann(n_bands: int = 2):
    union = FeatureUnion([
        (f"band_{i}", make_pipeline(BandSelect(i), Covariances(estimator="oas"), TangentSpace(metric="riemann")))
        for i in range(n_bands)
    ])
    return make_pipeline(union, LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[1 / 3, 1 / 3, 1 / 3]))


def eval_layout(channels: list[str], label: str):
    accs, fars = [], []
    for s in range(1, 11):
        X_cal, y_cal = load_band_epochs(s, [4, 8], channels)
        X_te, y_te = load_band_epochs(s, [12], channels)
        clf = make_fb_riemann(2)
        clf.fit(X_cal, y_cal)
        preds = clf.predict(X_te)
        accs.append(accuracy_score(y_te, preds))
        fars.append(float(np.mean(preds[y_te == 0] != 0)))
    print(f"  {label:32s} -> Mean Acc: {np.mean(accs)*100:5.1f}% | Mean Rest FAR: {np.mean(fars)*100:5.1f}% | S004: {accs[3]*100:5.1f}% | S007: {accs[6]*100:5.1f}%")


def main():
    print("=" * 78)
    print("HEADSET CHANNEL & FILTER-BANK COMPARISON (Runs 4+8 -> Unseen Run 12)")
    print("=" * 78)
    eval_layout(CH_4, "4ch Filter-Bank (Ganglion)")
    eval_layout(CH_8, "8ch Filter-Bank (Cyton)")


if __name__ == "__main__":
    main()

