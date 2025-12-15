# Guided Piano v2 — Adaptive Gesture‑Based Piano Learning

Guided Piano is a human‑computer interaction system that uses **computer vision** (hand tracking) + **gesture classification** to drive a **personalized, adaptive** piano learning experience. It includes three modes:
- **Tutorial Mode** — stepwise lessons with corrective feedback.
- **Free Play Mode** — real‑time gesture → note/chord mapping.
- **Challenge Mode** — timed patterns, scores, and progression.

This starter repo gives you a runnable webcam demo, a Streamlit UI skeleton, an adaptive feedback engine, and a trainable gesture classifier.

---

## Quickstart

```bash
# 1) Create & activate a virtual env (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Test the webcam hand‑tracking demo (press q to quit)
python src/demo_cli.py

# 4) Launch the Streamlit UI (skeleton)
streamlit run src/app.py
```

> Note: If `pygame.midi` fails (no MIDI device), the app will fall back to WAV playback via `simpleaudio`.

---

## Project Structure

```
guided_piano_v2/
├── assets/
│   └── piano_sounds/           # (optional) put WAV samples here
├── data/
│   ├── gestures/               # collected, labeled gesture samples
│   └── sessions/               # per‑user practice/session logs
├── models/                     # trained classifiers (.pkl)
├── src/
│   ├── app.py                  # Streamlit UI (modes)
│   ├── demo_cli.py             # OpenCV webcam demo
│   ├── core/
│   │   ├── hand_tracking.py    # MediaPipe Hands wrapper
│   │   ├── gesture_classifier.py
│   │   ├── sound_engine.py     # MIDI + WAV fallback
│   │   └── feedback_engine.py  # adaptive coaching + metrics
│   ├── modes/
│   │   ├── tutorial.py
│   │   ├── free_play.py
│   │   └── challenge.py
│   └── utils/
│       ├── features.py         # landmark → features
│       ├── mappings.py         # gesture→note/chord maps
│       └── storage.py          # simple persistence helpers
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Minimal Workflow

1. **Collect** gesture data in `data/gestures/` using the CLI demo (press `c` to capture; edit code to label).
2. **Extract features & train** a lightweight classifier (`src/core/gesture_classifier.py` CLI entry).
3. **Run Streamlit** and test Tutorial/Free/Challenge modes.
4. **Tune adaptation** rules in `feedback_engine.py` (tempo, next‑lesson logic, hints).
5. **Evaluate** against other apps (time‑to‑task, accuracy, SUS score).

---

## Next Milestones

- [ ] Add per‑gesture calibration screen.
- [ ] Implement posture hints (e.g., finger lift/bend angles).
- [ ] Persist per‑user model adaptively.
- [ ] Add XP/streaks, badges, and level unlocks.
- [ ] Export practice reports (PDF/CSV).

## Optional: cvzone demo
If you prefer using **cvzone** for quick finger→chord control (like Air‑Piano), run:

```bash
pip install cvzone
python src/demo_cvzone.py
```


### Backend Toggle
In the Streamlit sidebar, choose between **MediaPipe + Classifier** (trainable) and **cvzone (no-training)** backends. The cvzone backend maps raised fingers to D-major family chords for instant play and works with Tutorial/Challenge via a D-major curriculum.
