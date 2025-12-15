# --- path shim (must be first) ---
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ---------------------------------

import time
import streamlit as st
import numpy as np
import cv2

from src.core.hand_tracking import HandTracker
from src.core.gesture_classifier import GestureClassifier
from src.core.engine_cvzone import CvzoneDetector
from src.core.feedback_engine import AdaptiveCoach
from src.core.sound_engine import SoundEngine
from src.modes.tutorial import TutorialMode
from src.modes.free_play import FreePlayMode
from src.utils.features import extract_features

st.set_page_config(page_title="Guided Piano", layout="wide")
st.title("🎹 Guided Piano — Adaptive Gesture-Based Learning")
st.write("Webcam demo with **Tutorial** and **Free Play** modes. Click **Start/Stop** to begin.")

# Sidebar controls
mode = st.sidebar.radio("Mode", ["Tutorial", "Free Play"])
backend = st.sidebar.selectbox("Detection Backend", ["MediaPipe + Classifier", "cvzone (no-training)"])
start_btn = st.sidebar.button("Start/Stop")

# Reset & lock controls
if st.sidebar.button("↩ Reset to Level 1"):
    st.session_state.coach = AdaptiveCoach()
    st.session_state._tutorial_reset = True  # apply after state_mode is created

lock_lvl = st.sidebar.checkbox("Lock level at 1", value=False)

if "running" not in st.session_state:
    st.session_state.running = False
if start_btn:
    st.session_state.running = not st.session_state.running

# Tabs (mirrors your deck narrative)
live_tab, overview_tab, tech_tab = st.tabs(["Live Demo", "Overview", "Tech & Architecture"])

with overview_tab:
    st.subheader("HCI Focus")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "**Input**: Webcam hand gestures\n\n"
            "**Feedback**: On-video hints + chord audio\n\n"
            "**Adaptation**: Tempo/level from accuracy & reaction time"
        )
    with c2:
        st.markdown(
            "**Modes**\n"
            "- **Tutorial**: Next-target guidance, one-shot per step\n"
            "- **Free Play**: Open exploration"
        )

with tech_tab:
    st.subheader("Stack")
    st.markdown(
        "- **Vision**: MediaPipe / CvZone + OpenCV\n"
        "- **ML**: RandomForest (MediaPipe path)\n"
        "- **Sound**: pygame.midi → WAV fallback\n"
        "- **UI**: Streamlit\n"
        "- **Adaptive**: `AdaptiveCoach`"
    )

with live_tab:
    # Lazy init of engines
    if backend == "MediaPipe + Classifier":
        if "tracker" not in st.session_state:
            try:
                st.session_state.tracker = HandTracker()
            except Exception as e:
                st.error(f"HandTracker init failed: {e}")
    else:
        if "cvz" not in st.session_state:
            try:
                st.session_state.cvz = CvzoneDetector()
            except Exception as e:
                st.error(f"cvzone init failed: {e}")

    if "sound" not in st.session_state:
        st.session_state.sound = SoundEngine()
    if "coach" not in st.session_state:
        st.session_state.coach = AdaptiveCoach()

    # Classifier only for MediaPipe path
    clf = None
    if backend == "MediaPipe + Classifier":
        try:
            clf_try = GestureClassifier()
            clf_try.load("models/gesture_model.pkl")
            clf = clf_try
        except Exception:
            st.warning("No trained classifier found (models/gesture_model.pkl). Using a stub label.")

    # Mode wiring (NO Challenge)
    if mode == "Tutorial":
        if backend == "cvzone (no-training)":
            lesson = ["CHORD_D_MAJOR", "CHORD_E_MINOR", "CHORD_FSHARP_MINOR", "CHORD_G_MAJOR", "CHORD_A_MAJOR"]
            state_mode = TutorialMode(st.session_state.coach, st.session_state.sound, lesson=lesson)
        else:
            state_mode = TutorialMode(st.session_state.coach, st.session_state.sound)
    else:
        state_mode = FreePlayMode(st.session_state.sound)

    # Apply tutorial reset once if requested
    if st.session_state.get("_tutorial_reset"):
        if mode == "Tutorial":
            if hasattr(state_mode, "idx"):
                state_mode.idx = 0
            for name, value in [
                ("_await_release", False),
                ("_lock_label", None),
                ("_release_count", 0),
                ("_cooldown_until_ms", 0.0),
            ]:
                if hasattr(state_mode, name):
                    setattr(state_mode, name, value)
        st.session_state._tutorial_reset = False

    # Dashboard elements
    m1, m2, m3, m4 = st.columns(4)
    m1p, m2p, m3p, m4p = m1.empty(), m2.empty(), m3.empty(), m4.empty()
    next_target_box = st.markdown("")
    prog = st.progress(0)

    FRAME = st.image(np.zeros((360, 640, 3), dtype=np.uint8))

    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam.")
            st.session_state.running = False
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            last_ts = time.time()

            try:
                while st.session_state.running:
                    ok, frame = cap.read()
                    if not ok:
                        st.error("Frame grab failed.")
                        break

                    frame = cv2.flip(frame, 1)
                    out = frame.copy()

                    # Optional: freeze level at 1
                    if lock_lvl:
                        st.session_state.coach.level = 1
                        st.session_state.coach.tempo_bpm = 60

                    # === PRE-CHECK for green cue: compare BEFORE advancing ===
                    tgt_before = state_mode.target_label() if mode == "Tutorial" else None

                    if backend == "cvzone (no-training)":
                        label, conf, drawn = st.session_state.cvz.infer(out)
                        out = drawn
                        dt_ms = int((time.time() - last_ts) * 1000)
                        last_ts = time.time()

                        # did we match current target before it advances?
                        was_match = (mode == "Tutorial" and tgt_before is not None and label == tgt_before and conf >= 0.6)

                        # advance/update
                        if mode == "Free Play":
                            info = state_mode.handle_prediction(label, conf)
                        else:
                            info = state_mode.handle_prediction(label, conf, dt_ms)

                    else:
                        hands = st.session_state.tracker.process(out)
                        out = st.session_state.tracker.draw(out, hands)

                        label, conf = ("", 0.0)
                        if hands:
                            feats = extract_features(hands[0].points)["vector"]
                            if clf is None:
                                label, conf = ("NOTE_C4", 0.7)
                            else:
                                label, conf = clf.predict_label(feats)

                        dt_ms = int((time.time() - last_ts) * 1000)
                        last_ts = time.time()

                        was_match = (mode == "Tutorial" and tgt_before is not None and label == tgt_before and conf >= 0.6)

                        if mode == "Free Play":
                            info = state_mode.handle_prediction(label, conf)
                        else:
                            info = state_mode.handle_prediction(label, conf, dt_ms)

                    # --- Progress & next target (AFTER advancing) ---
                    if mode == "Tutorial" and hasattr(state_mode, "target_label"):
                        tgt_after = state_mode.target_label()
                        total = len(getattr(state_mode, "lesson", [])) or 1
                        idx = getattr(state_mode, "idx", 0)
                        if tgt_after is None:
                            next_target_box.markdown("**Next target:** ✅ Lesson complete!")
                            prog.progress(100)
                        else:
                            next_target_box.markdown(f"**Next target:** `{tgt_after}`")
                            prog.progress(int(100 * idx / total))

                    # --- Tutorial hint overlay using pre-check result ---
                    if mode == "Tutorial":
                        if info.get("done"):
                            cv2.rectangle(out, (10, 100), (630, 140), (0, 0, 0), -1)
                            cv2.putText(out, "Lesson complete! 🎉", (20, 132),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 180, 60), 2)
                        else:
                            tgt_display = tgt_before if tgt_before is not None else info.get("target", "")
                            color = (40, 180, 60) if was_match else (0, 0, 255)
                            msg = f"{'✓ Nice! ' if was_match else '✗ Try again! Play '} {tgt_display}"
                            cv2.rectangle(out, (10, 100), (630, 140), (0, 0, 0), -1)
                            cv2.putText(out, msg, (20, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    # Metrics tiles
                    coach = info.get("coach", {}) if isinstance(info, dict) else {}
                    if coach:
                        m1p.metric("Accuracy", f"{int(coach.get('accuracy', 0)*100)}%")
                        m2p.metric("Avg Reaction", f"{coach.get('avg_reaction_ms', 0)} ms")
                        m3p.metric("Tempo BPM", coach.get("tempo_bpm", 60))
                        m4p.metric("Level", coach.get("level", 1))

                    # Prediction banner
                    if label:
                        cv2.putText(out, f"Pred: {label} ({conf:.2f})", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                    FRAME.image(out, channels="BGR")
                    time.sleep(0.01)
            finally:
                cap.release()
    else:
        st.info("Click **Start/Stop** to toggle webcam.")
