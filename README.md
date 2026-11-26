# Brain–Computer Interface (BCI)

A step-by-step project to recognize simple imagined hand movements (`Rest`, `Left Hand`, `Right Hand`) from scalp EEG recordings and connect those predictions to physical hardware.

- **Interactive Visual Lessons:** [loyality7.github.io/BCI](https://loyality7.github.io/BCI/)

---

## What We Are Trying to Do (In Stages)

1. **Stage 1 — Learn & Decode Offline (`Done`)**  
   Clean noisy human EEG recordings (`8–30 Hz`), verify motor-cortex signal shifts (`C3` vs. `C4`), benchmark statistical and deep-learning models across public datasets, and export a fast, calibrated classifier (`< 10 ms` per window, `0%` Rest misfires on calibrated subjects).
2. **Stage 2 — Real-Time Simulation & Hardware Control (`Done`)**  
   Replays unseen EEG at `160 Hz` with causal `8–30 Hz` filtering, a `150 µV` artifact guard, `3-hit` debounce, and `2.0 s` cooldown to drive a physical Arduino LED over USB Serial (`COM5`, `6.34 ms` mean latency).
3. **Stage 3 — Live EEG Headset (`Ready for Hardware`)**  
   Plug-and-play BrainFlow runner (`live/brainflow_runner.py`) supporting `synthetic`, `ganglion`, and `cyton` boards with 5-minute guided calibration.
4. **Stage 4 — Hands-Free Device / Rover Control**  
   Scale the single ESP32/Arduino LED into a real physical controller—steering a 2-wheel rover (`Left`, `Right`, `Rest` = stop) or toggling room appliances via smart relays.

---

## Repository Layout

- **`learn/`** — Visual reference lessons (`01`–`06`) deployed to [GitHub Pages](https://loyality7.github.io/BCI/).
- **`research/`** — Stage 1 scripts and the [research log](research/RESEARCH.md) with real-data plots and numbers.
- **`models/`** — Reusable training, calibration, and inference module ([stage1_classifier.py](models/stage1_classifier.py)).
- **`simulation/`** — `160 Hz` causal replay, parameter sweeps, and 4-ch vs. 8-ch hardware benchmarks.
- **`hardware/`** — Arduino/ESP32 serial LED controller ([led_controller.ino](hardware/led_controller/led_controller.ino)).
- **`live/`** — Live WebSocket SVG/Canvas simulator ([simulator_server.py](live/simulator_server.py)), Streamlit monitor, and Stage 3 BrainFlow runner ([brainflow_runner.py](live/brainflow_runner.py)).
- **`results/`** — Generated EEG traces, frequency spectra, confusion matrices, and Stage 2 replay plots.

---

## Stage 1 & 2 Results Summary

- **Winning Pipeline:** `PyRiemann` (`OAS Covariance → Riemannian TangentSpace → Shrinkage LDA`, 64 channels, `8–30 Hz`)
- **Multi-Dataset Accuracy:**
  - `BNCI2014_001` (4-class, 9 subjects): **`72.2%`** *(chance `25.0%`)*
  - `BNCI2014_004` (2-class, 3 channels, 9 subjects): **`74.1%`** *(chance `50.0%`)*
  - `PhysionetMI` (3-class, 10 subjects): **`63.4%`** mean accuracy, **`5.1%`** Rest False Activation Rate
- **Calibrated Readiness (`Runs 4+8 → Unseen Run 12`, Subject `004`):** `80.0%` accuracy, `0.0%` Rest misfires, `6.08 ms` latency.
- **Stage 2 Live Hardware (`Subject 007`, Unseen `Run 12 → Arduino COM5`):** `10 / 10` cue-aligned LED switches, `0` Rest misfires, `6.34 ms` mean end-to-end latency.

---

## Quickstart

```bash
# 1. Create Python 3.11 environment and install dependencies
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe mne moabb pyriemann scikit-learn scipy matplotlib torch braindecode brainflow pyserial starlette uvicorn websockets "setuptools<81"

# 2. Download public EEG datasets into datasets/
.venv/Scripts/python.exe research/download_datasets.py

# 3. Run Stage 1 verification & export models
.venv/Scripts/python.exe research/06_export_and_verify_stage1.py

# 4. Run Stage 2 live visual simulator (open http://127.0.0.1:8000)
.venv/Scripts/python.exe live/simulator_server.py
```
