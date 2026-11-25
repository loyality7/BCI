"""
BCI — Stage 1 Dataset Downloader
=====================================
Downloads and caches public motor-imagery EEG datasets into:
    d:\\Projects\\BCI\\datasets\\

Usage:
    python research/download_datasets.py          # Starter cohorts across 6 datasets
    python research/download_datasets.py --all    # Full subjects + large cohorts
"""

import argparse
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "datasets"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Force MNE & MOABB to store all data inside d:\Projects\BCI\datasets
os.environ["MNE_DATA"] = str(DATA_DIR)
os.environ["MNE_DATASETS_EEGBCI_PATH"] = str(DATA_DIR / "physionet")

import mne
from moabb.datasets import (
    BNCI2014_001,
    BNCI2014_004,
    Cho2017,
    Kaya2018,
    Lee2019,
    PhysionetMI,
    Schirrmeister2017,
    Zhou2016,
)


def download_dataset(name: str, dataset, subjects: list[int] | None) -> None:
    """Download specified subjects (or all if None) for a MOABB dataset."""
    target = subjects if subjects is not None else dataset.subject_list
    print(f"\n[{name}] Downloading {len(target)} subject(s) -> {DATA_DIR}")
    try:
        paths = dataset.data_path(subject=target[0], path=str(DATA_DIR))
        # Download remaining subjects one by one so progress is visible and resumable
        for idx, subj in enumerate(target, start=1):
            dataset.download(subject_list=[subj], path=str(DATA_DIR), verbose=False)
            print(f"  [{name}] {idx}/{len(target)} (subject {subj}) ready")
    except Exception as exc:
        print(f"  [{name}] Warning: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Stage 1 EEG datasets into datasets/")
    parser.add_argument("--all", action="store_true", help="Download all subjects including large cohorts")
    args = parser.parse_args()

    mne.set_log_level("WARNING")
    mne.set_config("MNE_DATA", str(DATA_DIR), set_env=True)

    # Compact, complete core benchmarks (always download all subjects)
    plan = [
        ("BNCI2014_001", BNCI2014_001(), None),
        ("BNCI2014_004", BNCI2014_004(), None),
        ("Zhou2016", Zhou2016(), None),
        ("PhysionetMI", PhysionetMI(), None if args.all else list(range(1, 11))),
        ("Kaya2018", Kaya2018(), None if args.all else [1, 2, 3]),
    ]

    if args.all:
        plan.extend([
            ("Cho2017", Cho2017(), None),
            ("Lee2019", Lee2019(), None),
            ("Schirrmeister2017", Schirrmeister2017(), None),
        ])

    print("=" * 68)
    print(f"BCI DATASET DOWNLOADER | MODE: {'FULL' if args.all else 'STARTER COHORTS'}")
    print(f"Target directory: {DATA_DIR}")
    print("=" * 68)

    for name, ds, subjs in plan:
        download_dataset(name, ds, subjs)

    print("\nDone. All downloaded datasets are cached in:", DATA_DIR)


if __name__ == "__main__":
    main()

