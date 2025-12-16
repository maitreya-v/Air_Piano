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

## Repository Structure (Typical)
> (Update names if your folders differ)

