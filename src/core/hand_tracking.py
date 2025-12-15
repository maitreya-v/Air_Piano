from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
import cv2

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    print("Warning: mediapipe not available. Install with `pip install mediapipe`.")

@dataclass
class HandLandmarks:
    # 21 landmarks, each as (x, y) in image-normalized coords [0,1]
    points: np.ndarray  # shape (21, 2)
    handedness: str     # 'Left' or 'Right'
    score: float

class HandTracker:
    def __init__(self, max_hands: int = 1, detection_conf: float = 0.5, tracking_conf: float = 0.5):
        if mp is None:
            raise ImportError("mediapipe is required for HandTracker.")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles

    def process(self, frame_bgr: np.ndarray) -> List[HandLandmarks]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(frame_rgb)
        out: List[HandLandmarks] = []
        if res.multi_hand_landmarks and res.multi_handedness:
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                pts = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)
                handedness = hd.classification[0].label
                score = hd.classification[0].score
                out.append(HandLandmarks(points=pts, handedness=handedness, score=score))
        return out

    def draw(self, frame_bgr: np.ndarray, landmarks: List[HandLandmarks]) -> np.ndarray:
        if mp is None:
            return frame_bgr
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        for lm in landmarks:
            # Build a faux mediapipe landmark object for drawing
            lm_obj = self.mp_hands.HandLandmark
            # We can use drawing utils only with the original results; instead, draw basic circles/lines here.
            for (x, y) in lm.points:
                cv2.circle(frame_bgr, (int(x * frame_bgr.shape[1]), int(y * frame_bgr.shape[0])), 3, (0, 255, 0), -1)
            cv2.putText(frame_bgr, f"{lm.handedness} ({lm.score:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame_bgr