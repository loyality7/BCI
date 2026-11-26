"""
BCI — Stage 3 Ready: Live Headset Calibration & Arduino Control (BrainFlow)
===========================================================================
Plug-and-play runner for real EEG headsets (OpenBCI Ganglion/Cyton, etc.)
and BrainFlow's SYNTHETIC_BOARD for end-to-end testing before hardware arrives.

Flow:
  1. Connect to BrainFlow board (synthetic, ganglion, cyton)
  2. Guided Calibration: prompts Rest / Left / Right trials -> fits PyRiemann (<1s)
  3. Live Loop: causal 8-30 Hz filter -> 150 uV artifact guard -> 3-hit debounce
                -> 2.0s cooldown -> USB Serial ('1'/'0') to Arduino on COM5

Usage:
  .venv/Scripts/python.exe live/brainflow_runner.py --board synthetic --trials 4 --port COM5
  .venv/Scripts/python.exe live/brainflow_runner.py --board ganglion --serial-port COM4 --port COM5
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from scipy.signal import butter, sosfilt, sosfilt_zi

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.stage1_classifier import build_pipeline, predict_window, save_model

BOARD_MAP = {
    "synthetic": BoardIds.SYNTHETIC_BOARD.value,
    "ganglion": BoardIds.GANGLION_BOARD.value,
    "cyton": BoardIds.CYTON_BOARD.value,
}


def collect_calibration_trials(board: BoardShim, eeg_chs: list[int], sfreq: int, n_trials_per_class: int, trial_sec: float = 3.0):
    """Guide user through Rest / Left / Right trials and collect filtered windows."""
    sos = butter(4, [8.0, 30.0], btype="bandpass", fs=sfreq, output="sos")
    tasks = [("Rest (relax hands)", 0), ("Imagine LEFT hand", 1), ("Imagine RIGHT hand", 2)]
    X_list, y_list = [], []

    print("\n--- CALIBRATION PHASE ---")
    for round_idx in range(n_trials_per_class):
        for prompt, cls_id in tasks:
            print(f"  [{round_idx + 1}/{n_trials_per_class}] {prompt} for {trial_sec:.1f} s...")
            board.get_board_data()  # flush old samples
            time.sleep(trial_sec + 0.2)
            data = board.get_board_data()
            eeg = data[eeg_chs, :]
            n_req = int(trial_sec * sfreq)
            if eeg.shape[1] >= n_req:
                win = eeg[:, -n_req:] * 1e-6  # uV -> V (matches MNE/Stage 1 scale)
                win_filt = sosfilt(sos, win, axis=1)
                X_list.append(win_filt)
                y_list.append(cls_id)

    return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=int)


def main():
    parser = argparse.ArgumentParser(description="Stage 3 Live Headset Calibration & Arduino Control")
    parser.add_argument("--board", choices=list(BOARD_MAP.keys()), default="synthetic")
    parser.add_argument("--serial-port", type=str, default="", help="EEG dongle COM port (for OpenBCI)")
    parser.add_argument("--port", type=str, default=None, help="Arduino/ESP32 COM port (e.g. COM5)")
    parser.add_argument("--trials", type=int, default=4, help="Calibration trials per class")
    parser.add_argument("--duration", type=float, default=15.0, help="Live loop duration in seconds")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--debounce", type=int, default=3)
    parser.add_argument("--artifact-uv", type=float, default=150.0)
    parser.add_argument("--cooldown", type=float, default=2.0)
    args = parser.parse_args()

    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    if args.serial_port:
        params.serial_port = args.serial_port

    board_id = BOARD_MAP[args.board]
    board = BoardShim(board_id, params)
    sfreq = BoardShim.get_sampling_rate(board_id)
    eeg_chs = BoardShim.get_eeg_channels(board_id)

    ser = None
    if args.port:
        import serial
        ser = serial.Serial(args.port, 115200, timeout=0.1)
        time.sleep(3.0)
        print(f"Arduino connected on {args.port}")

    board.prepare_session()
    board.start_stream()
    print(f"Started BrainFlow stream: {args.board} ({len(eeg_chs)} EEG ch @ {sfreq} Hz)")

    try:
        X_cal, y_cal = collect_calibration_trials(board, eeg_chs, sfreq, args.trials, trial_sec=3.0)
        model = build_pipeline()
        model.fit(X_cal, y_cal)
        save_model(model, ROOT_DIR / "models" / f"pyriemann_{args.board}.joblib", metadata={"board": args.board, "sfreq": sfreq})
        print(f"Calibrated PyRiemann on {len(X_cal)} live trials -> models/pyriemann_{args.board}.joblib")

        # Live closed-loop inference
        win_samples = int(3.0 * sfreq)
        sos = butter(4, [8.0, 30.0], btype="bandpass", fs=sfreq, output="sos")
        zi = np.repeat(sosfilt_zi(sos)[:, np.newaxis, :], len(eeg_chs), axis=1)
        buffer = np.zeros((len(eeg_chs), win_samples), dtype=np.float64)

        led_on, streak_cls, streak_count, last_switch_t = 0, 0, 0, -1e9
        t_start = time.perf_counter()
        print(f"\n--- LIVE MODE ({args.duration:.0f} s) ---")
        board.get_board_data()  # flush

        while (time.perf_counter() - t_start) < args.duration:
            time.sleep(0.25)
            new_data = board.get_board_data()
            if new_data.shape[1] == 0:
                continue

            chunk_v = new_data[eeg_chs, :] * 1e-6
            filt_chunk, zi = sosfilt(sos, chunk_v, axis=1, zi=zi)
            n_new = min(filt_chunk.shape[1], win_samples)
            buffer = np.roll(buffer, -n_new, axis=1)
            buffer[:, -n_new:] = filt_chunk[:, -n_new:]

            peak_uv = float(np.max(np.abs(buffer)) * 1e6)
            cur_t = time.perf_counter() - t_start

            if peak_uv > args.artifact_uv:
                streak_cls, streak_count = 0, 0
                print(f"  t={cur_t:5.2f}s | ARTIFACT GUARD ({peak_uv:.0f} uV) -> Hold")
                continue

            res = predict_window(model, buffer, active_threshold=args.threshold)
            pred = res["class_id"]

            if pred in (1, 2):
                streak_count = streak_count + 1 if pred == streak_cls else 1
                streak_cls = pred
            else:
                streak_cls, streak_count = 0, 0

            if streak_count >= args.debounce and (cur_t - last_switch_t) >= args.cooldown:
                new_led = 1 if streak_cls == 1 else 0
                if new_led != led_on:
                    led_on = new_led
                    last_switch_t = cur_t
                    if ser is not None:
                        ser.write(b"1\n" if led_on else b"0\n")
                    print(f"  t={cur_t:5.2f}s | {res['label']:5s} -> LED {'ON' if led_on else 'OFF'} ({res['latency_ms']:.2f} ms)")

    finally:
        board.stop_stream()
        board.release_session()
        if ser is not None:
            ser.close()
        print("Session closed cleanly.")


if __name__ == "__main__":
    main()

