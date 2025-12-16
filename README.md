# Guided Piano 🎹 — Adaptive Gesture-Based Learning (Webcam-Only)

Guided Piano is a webcam-based HCI prototype that teaches **piano chord concepts** using **mid-air hand gestures**. It tracks hand landmarks in real time, classifies chord-like poses, provides **visual + audio feedback**, and **adapts tempo/difficulty** based on short-term performance.

---

## Demo (What it does)
- **Tutorial mode:** shows a target chord → you perform the gesture → system confirms success + plays chord audio
- **Free Play mode:** no target → plays whichever chord is recognized (exploration)
- **Adaptation:** adjusts tempo/level using recent **accuracy** and **reaction time**

---

## Key Features
- ✅ Webcam-only input (no special hardware)
- ✅ Real-time hand landmark tracking (MediaPipe / CvZone backend)
- ✅ Gesture/chord classification (Random Forest path supported)
- ✅ Temporal smoothing / voting to reduce label flicker
- ✅ Multimodal feedback: UI overlays + chord audio
- ✅ AdaptiveCoach: tempo + level progression using rolling performance

---

## Tech Stack
- **UI:** Streamlit
- **Vision:** MediaPipe Hands / CvZone + OpenCV
- **ML:** RandomForest (scikit-learn) for chord labels (MediaPipe pipeline)
- **Audio:** `pygame.midi` → WAV fallback
- **Adaptive logic:** AdaptiveCoach module (rolling accuracy/RT)

---

## System Pipeline (High-Level)
1. **Webcam** frames captured in real time  
2. **Hand landmarks** detected per frame  
3. **Features** computed (normalized coordinates + optional geometry features)  
4. **Classifier** predicts chord label + confidence  
5. **Temporal voting** stabilizes labels (reduces flicker)  
6. **Feedback** renders overlays + plays chord audio  
7. **Adaptation** adjusts tempo/level based on recent performance  

---

## Repository Structure

This folder contains the core computer vision, gesture recognition, and feedback modules that power the system.

hand_tracking.py
Handles real-time hand detection and landmark extraction using webcam input.

gesture_classifier.py
Converts extracted hand features into discrete gesture labels using rule-based or learned mappings.

engine_cvzone.py
Alternative CV backend using CvZone for robust hand tracking and visualization.

feedback_engine.py
Implements adaptive feedback logic (visual/audio cues) based on user performance and errors.

sound_engine.py
Responsible for playing audio feedback and system sounds.

🎮 src/modes/ — Interaction Modes

Each file in this folder represents a distinct interaction mode, designed to study different learning and exploration behaviors.

tutorial.py
Guided mode that introduces gestures step-by-step with explicit feedback and instructions.

free_play.py
Open-ended mode allowing users to experiment freely without constraints.

challenge.py
Task-based mode that evaluates user performance under time or accuracy constraints.

🧩 src/utils/ — Shared Utilities

This folder contains reusable helper modules used across the system.

features.py
Extracts geometric and temporal features from hand landmarks.

mappings.py
Defines mappings between gestures, actions, and system responses.

storage.py
Handles logging, saving user performance data, and session storage (if enabled).

🚀 Entry Points & Demos

app.py
Main application entry point (used for running the full system).

demo_cli.py
Command-line demo for testing gesture recognition logic without UI.

demo_cvzone.py
Visual demo using CvZone to showcase real-time hand tracking and gesture detection.

📦 Other Files

requirements.txt
Lists all Python dependencies required to run the project.

README.md
Project overview, setup instructions, and documentation.

LICENSE
Open-source license for the project.

