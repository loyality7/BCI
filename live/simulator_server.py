"""
BCI — Stage 2/3 Real-Time WebSocket Simulator Server
====================================================
Serves the streamlined SVG/Canvas simulator (live/index.html) and streams
real Subject 004 / 007 unseen Run 12 EEG over WebSocket with causal 8-30 Hz
filtering, 150 uV artifact guard, 3-hit debounce, and optional COM5 Arduino.

Run:
    .venv/Scripts/python.exe live/simulator_server.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci
from scipy.signal import butter, sosfilt, sosfilt_zi
from starlette.applications import Starlette
from starlette.responses import FileResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect
import uvicorn

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.stage1_classifier import load_model, predict_window

DATA_DIR = ROOT_DIR / "datasets" / "physionet"
HTML_PATH = Path(__file__).resolve().parent / "index.html"
SFREQ = 160.0
WINDOW_SAMPLES = int(3.0 * SFREQ)
STEP_SAMPLES = int(0.25 * SFREQ)
LABELS = ["Rest", "Left", "Right"]


def load_subject_run12(subject: int):
    mne.set_log_level("WARNING")
    edf_path = eegbci.load_data(subject, [12], path=str(DATA_DIR), update_path=False)[0]
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    eegbci.standardize(raw)
    raw_data = raw.get_data()

    truth = np.zeros(raw.n_times, dtype=int)
    for onset, dur, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        if desc in {"T0": 0, "T1": 1, "T2": 2}:
            s0 = int(onset * SFREQ)
            s1 = min(raw.n_times, int((onset + dur) * SFREQ))
            truth[s0:s1] = {"T0": 0, "T1": 1, "T2": 2}[desc]

    idx_map = {ch: raw.ch_names.index(ch) for ch in ["C3", "Cz", "C4", "Pz"] if ch in raw.ch_names}
    model, _ = load_model(ROOT_DIR / "models" / f"pyriemann_s{subject:03d}.joblib")
    return raw_data, truth, idx_map, model


async def homepage(request):
    return FileResponse(HTML_PATH)


async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    ser = None
    try:
        cfg_raw = await websocket.receive_text()
        cfg = json.loads(cfg_raw)
        subject = int(cfg.get("subject", 7))
        speed = float(cfg.get("speed", 1.5))
        threshold = float(cfg.get("threshold", 0.60))
        debounce = int(cfg.get("debounce", 3))
        artifact_uv = float(cfg.get("artifact_uv", 150.0))
        com_port = cfg.get("port", "").strip()

        raw_data, truth, idx_map, model = load_subject_run12(subject)

        if com_port:
            import serial
            ser = serial.Serial(com_port, 115200, timeout=0.1)
            await asyncio.sleep(3.0)

        sos = butter(4, [8.0, 30.0], btype="bandpass", fs=SFREQ, output="sos")
        zi = np.repeat(sosfilt_zi(sos)[:, np.newaxis, :], raw_data.shape[0], axis=1) * raw_data[:, 0][np.newaxis, :, np.newaxis]

        buffer = np.zeros((raw_data.shape[0], WINDOW_SAMPLES), dtype=np.float64)
        samples_seen, led_on, streak_cls, streak_count, last_switch_t = 0, 0, 0, 0, -1e9

        for start in range(0, raw_data.shape[1] - STEP_SAMPLES + 1, STEP_SAMPLES):
            # Non-blocking check if client asked to stop
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                if msg == "STOP":
                    break
            except asyncio.TimeoutError:
                pass

            t0 = time.perf_counter()
            chunk = raw_data[:, start : start + STEP_SAMPLES]
            filt_chunk, zi = sosfilt(sos, chunk, axis=1, zi=zi)
            buffer = np.roll(buffer, -STEP_SAMPLES, axis=1)
            buffer[:, -STEP_SAMPLES:] = filt_chunk
            samples_seen += STEP_SAMPLES

            if samples_seen >= WINDOW_SAMPLES:
                cur_t = (start + STEP_SAMPLES) / SFREQ
                true_cls = int(truth[start + STEP_SAMPLES - 1])
                peak_uv = float(np.max(np.abs(buffer)) * 1e6)
                c3_i, cz_i, c4_i, pz_i = idx_map["C3"], idx_map["Cz"], idx_map["C4"], idx_map["Pz"]
                rms_uv = float(np.sqrt(np.mean(buffer[[c3_i, c4_i]] ** 2)) * 1e6)

                artifact_blocked = peak_uv > artifact_uv
                tx_pulse = False
                in_cooldown = (cur_t - last_switch_t) < 2.0

                if artifact_blocked:
                    pred_cls = 0
                    pred_label = "Hold"
                    probs = {"Rest": 1.0, "Left": 0.0, "Right": 0.0}
                    streak_cls, streak_count = 0, 0
                else:
                    res = predict_window(model, buffer, active_threshold=threshold)
                    pred_cls = res["class_id"]
                    pred_label = res["label"]
                    probs = res["probabilities"]

                    if pred_cls in (1, 2):
                        streak_count = streak_count + 1 if pred_cls == streak_cls else 1
                        streak_cls = pred_cls
                    else:
                        streak_cls, streak_count = 0, 0

                    if streak_count >= debounce and not in_cooldown:
                        new_led = 1 if streak_cls == 1 else 0
                        if new_led != led_on:
                            led_on = new_led
                            last_switch_t = cur_t
                            tx_pulse = True
                            if ser is not None:
                                ser.write(b"1\n" if led_on else b"0\n")

                lat_ms = (time.perf_counter() - t0) * 1000.0
                # Channel band power (uV^2) for live head visualization
                ch_power = {
                    ch: round(float(np.var(buffer[idx] * 1e6)), 1)
                    for ch, idx in idx_map.items()
                }

                payload = {
                    "t": round(cur_t, 2),
                    "true_label": LABELS[true_cls],
                    "pred_label": pred_label,
                    "probs": probs,
                    "led_on": led_on,
                    "tx_pulse": tx_pulse,
                    "artifact_blocked": artifact_blocked,
                    "in_cooldown": in_cooldown,
                    "streak": streak_count,
                    "debounce": debounce,
                    "threshold": threshold,
                    "peak_uv": round(peak_uv, 1),
                    "rms_uv": round(rms_uv, 1),
                    "latency_ms": round(lat_ms, 2),
                    "ch_power": ch_power,
                    "c3_wave": np.round(filt_chunk[c3_i] * 1e6, 2).tolist(),
                    "cz_wave": np.round(filt_chunk[cz_i] * 1e6, 2).tolist(),
                    "c4_wave": np.round(filt_chunk[c4_i] * 1e6, 2).tolist(),
                }
                await websocket.send_text(json.dumps(payload))

            elapsed = time.perf_counter() - t0
            target_dt = STEP_SAMPLES / (SFREQ * speed)
            if elapsed < target_dt:
                await asyncio.sleep(target_dt - elapsed)

        await websocket.send_text(json.dumps({"done": True}))
    except WebSocketDisconnect:
        pass
    finally:
        if ser is not None:
            ser.close()


app = Starlette(
    routes=[
        Route("/", homepage),
        WebSocketRoute("/ws", ws_stream),
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

