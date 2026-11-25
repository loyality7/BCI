# Non-Invasive Motor-Imagery BCI

A step-by-step research and engineering project to decode human brain activity (`Rest`, `Imagined Left Hand`, `Imagined Right Hand`) from non-invasive EEG signals and connect those predictions to physical hardware.

---

## What We Are Trying to Do (In Stages)

1. **Stage 1 — Learn & Decode Offline (`Done`)**  
   Clean noisy human EEG recordings (`8–30 Hz`), verify motor-cortex signal shifts (`C3` vs. `C4`), benchmark statistical and deep-learning models across public datasets, and export a fast, calibrated classifier (`< 10 ms` per window, `0%` Rest misfires on calibrated subjects).
2. **Stage 2 — Real-Time Simulation & Hardware Control (`Next`)**  
   Replay recorded EEG at its native `160 Hz` speed with causal filtering and a safety gate, sending live `ON` / `OFF` commands to an ESP32 microcontroller.
3. **Stage 3 — Live EEG Headset**  
   Replace file replay with a live consumer/open-source EEG headset and calibrate the decoder on personal recordings.
4. **Stage 4 — AI Assistant Triggering**  
   Use reliable 2–3 state brain commands as hands-free selection/confirmation triggers for an AI assistant.

---

## Repository Layout

- **`learn/`** — Visual reference lessons covering brain anatomy, neural oscillations, BCI fundamentals, system architecture, current clinical status, and a glossary ([01](learn/01_the_brain.html) → [06](learn/06_research_glossary.html)).
- **`research/`** — Stage 1 scripts and the [research log](research/RESEARCH.md) with real-data plots and numbers.
- **`models/`** — Reusable training, calibration, and inference module ([stage1_classifier.py](models/stage1_classifier.py)).
- **`results/`** — Generated EEG traces, frequency spectra, confusion matrices, and multi-dataset benchmarks.

---

## Stage 1 Results Summary

- **Winning Pipeline:** `PyRiemann` (`OAS Covariance → Riemannian TangentSpace → Shrinkage LDA`, 64 channels, `8–30 Hz`)
- **Multi-Dataset Accuracy:**
  - `BNCI2014_001` (4-class, 9 subjects): **`72.2%`** *(chance `25.0%`)*
  - `BNCI2014_004` (2-class, 3 channels, 9 subjects): **`74.1%`** *(chance `50.0%`)*
  - `PhysionetMI` (3-class, 10 subjects): **`63.4%`** mean accuracy, **`5.1%`** Rest False Activation Rate
- **Calibrated Readiness (`Runs 4+8 → Unseen Run 12`, Subject `004`):** `80.0%` accuracy, `0.0%` Rest misfires, `6.08 ms` latency.

---

## Quickstart

```bash
# 1. Create Python 3.11 environment and install dependencies
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe mne moabb pyriemann scikit-learn scipy matplotlib torch braindecode

# 2. Download public EEG datasets into datasets/
.venv/Scripts/python.exe research/download_datasets.py

# 3. Run Stage 1 inspection, filtering, baselines, and model export
.venv/Scripts/python.exe research/01_explore_subject.py
.venv/Scripts/python.exe research/02_filter_and_motor_power.py
.venv/Scripts/python.exe research/03_baseline_classifiers.py
.venv/Scripts/python.exe research/06_export_and_verify_stage1.py
```
