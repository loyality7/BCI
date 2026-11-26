"""
BCI — Live Signal Quality, Calibration & Hardware Control Dashboard
===================================================================
Run:
    .venv/Scripts/streamlit.exe run live/dashboard.py
"""

import sys
import time
from pathlib import Path

import mne
import numpy as np
import streamlit as st
from mne.datasets import eegbci
from scipy.signal import butter, sosfilt, sosfilt_zi

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.stage1_classifier import load_model, predict_window

DATA_DIR = ROOT_DIR / "datasets" / "physionet"
SFREQ = 160.0
WINDOW_SAMPLES = int(3.0 * SFREQ)
STEP_SAMPLES = int(0.25 * SFREQ)

st.set_page_config(page_title="BCI Live Monitor", layout="wide")
st.title("BCI Live Monitor & Hardware Control")
st.caption("Stage 2 Real-Time Causal Replay + Signal Quality Guard + Arduino Serial Control")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    subject = st.selectbox("Subject (Unseen Run 12)", [4, 7, 1], index=0)
    speed = st.slider("Replay Speed", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
    threshold = st.slider("Confidence Gate (Left/Right)", 0.40, 0.85, 0.60, 0.05)
    debounce = st.slider("Debounce Hits Required", 1, 5, 3, 1)
    artifact_uv = st.slider("Artifact Spike Guard (uV)", 80, 250, 150, 10)
    com_port = st.text_input("Arduino COM Port (optional, e.g. COM5)", value="")
    start_btn = st.button("Start Live Replay", type="primary", use_container_width=True)

# Layout placeholders
col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
m_state = col_stat1.empty()
m_led = col_stat2.empty()
m_contact = col_stat3.empty()
m_latency = col_stat4.empty()

col_left, col_right = st.columns([2, 1])
wave_chart = col_left.empty()
prob_chart = col_right.empty()
log_box = st.empty()

if not start_btn:
    st.info("Configure settings on the left and click **Start Live Replay** to watch real-time 8–30 Hz EEG, signal contact quality, probabilities, and LED state.")
else:
    mne.set_log_level("WARNING")
    edf_path = eegbci.load_data(subject, [12], path=str(DATA_DIR), update_path=False)[0]
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    eegbci.standardize(raw)
    raw_data = raw.get_data()
    c3_idx = raw.ch_names.index("C3")
    c4_idx = raw.ch_names.index("C4")

    model, _ = load_model(ROOT_DIR / "models" / f"pyriemann_s{subject:03d}.joblib")

    sos = butter(4, [8.0, 30.0], btype="bandpass", fs=SFREQ, output="sos")
    zi = np.repeat(sosfilt_zi(sos)[:, np.newaxis, :], raw_data.shape[0], axis=1) * raw_data[:, 0][np.newaxis, :, np.newaxis]

    ser = None
    if com_port.strip():
        import serial
        ser = serial.Serial(com_port.strip(), 115200, timeout=0.1)
        time.sleep(3.0)

    buffer = np.zeros((raw_data.shape[0], WINDOW_SAMPLES), dtype=np.float64)
    samples_seen, led_on, streak_cls, streak_count, last_switch_t = 0, 0, 0, 0, -1e9
    events_log = []

    try:
        for start in range(0, raw_data.shape[1] - STEP_SAMPLES + 1, STEP_SAMPLES):
            t0 = time.perf_counter()
            chunk = raw_data[:, start : start + STEP_SAMPLES]
            filt_chunk, zi = sosfilt(sos, chunk, axis=1, zi=zi)
            buffer = np.roll(buffer, -STEP_SAMPLES, axis=1)
            buffer[:, -STEP_SAMPLES:] = filt_chunk
            samples_seen += STEP_SAMPLES

            if samples_seen >= WINDOW_SAMPLES:
                cur_t = (start + STEP_SAMPLES) / SFREQ
                peak_uv = float(np.max(np.abs(buffer)) * 1e6)
                rms_uv = float(np.sqrt(np.mean(buffer[[c3_idx, c4_idx]] ** 2)) * 1e6)

                if peak_uv > artifact_uv:
                    label_str = "Artifact Hold"
                    probs = {"Rest": 1.0, "Left": 0.0, "Right": 0.0}
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    contact_status = f"SPIKE ({peak_uv:.0f} uV)"
                    streak_cls, streak_count = 0, 0
                else:
                    res = predict_window(model, buffer, active_threshold=threshold)
                    pred = res["class_id"]
                    label_str = res["label"]
                    probs = res["probabilities"]
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    contact_status = f"GOOD ({rms_uv:.1f} uV RMS)"

                    if pred in (1, 2):
                        streak_count = streak_count + 1 if pred == streak_cls else 1
                        streak_cls = pred
                    else:
                        streak_cls, streak_count = 0, 0

                    if streak_count >= debounce and (cur_t - last_switch_t) >= 2.0:
                        new_led = 1 if streak_cls == 1 else 0
                        if new_led != led_on:
                            led_on = new_led
                            last_switch_t = cur_t
                            if ser is not None:
                                ser.write(b"1\n" if led_on else b"0\n")
                            events_log.append(f"t={cur_t:5.2f}s | {label_str} -> LED {'ON' if led_on else 'OFF'}")

                m_state.metric("Predicted State", label_str)
                m_led.metric("Hardware LED", "ON" if led_on else "OFF")
                m_contact.metric("Signal Quality (C3/C4)", contact_status)
                m_latency.metric("Latency", f"{lat_ms:.1f} ms")

                # Show last 2 seconds of C3 & C4 in uV
                wave_df = {
                    "C3 (Left Motor uV)": buffer[c3_idx, -320:] * 1e6,
                    "C4 (Right Motor uV)": buffer[c4_idx, -320:] * 1e6,
                }
                wave_chart.line_chart(wave_df, height=260)
                prob_chart.bar_chart({"Probability": [probs["Rest"], probs["Left"], probs["Right"]]}, height=260)
                if events_log:
                    log_box.code("\n".join(events_log[-8:]))

            elapsed = time.perf_counter() - t0
            target_dt = STEP_SAMPLES / (SFREQ * speed)
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
    finally:
        if ser is not None:
            ser.close()
        st.success("Replay complete.")

