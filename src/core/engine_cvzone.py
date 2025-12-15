from typing import Tuple, Set, Optional
import cv2
try:
    from cvzone.HandTrackingModule import HandDetector
except Exception as e:
    HandDetector = None

# Map finger index -> chord label (consistent with utils/mappings.py)
FINGER_TO_LABEL = {
    0: "CHORD_D_MAJOR",     # Thumb
    1: "CHORD_E_MINOR",     # Index
    2: "CHORD_FSHARP_MINOR",# Middle
    3: "CHORD_G_MAJOR",     # Ring
    4: "CHORD_A_MAJOR",     # Pinky
}

class CvzoneDetector:
    def __init__(self, detection_conf: float = 0.7, max_hands: int = 2):
        if HandDetector is None:
            raise ImportError("cvzone is required. Install with `pip install cvzone`.")
        self.detector = HandDetector(detectionCon=detection_conf, maxHands=max_hands)

    def infer(self, frame_bgr) -> Tuple[str, float, any]:
        """Return (label, confidence, drawn_frame). Picks the first raised finger if multiple are up."""
        img = frame_bgr.copy()
        hands, img = self.detector.findHands(img, draw=True)
        label = "NONE"; conf = 0.0
        raised: Set[int] = set()
        if hands:
            # cvzone returns list of hand dicts; fingersUp returns list of 0/1 for [thumb..pinky]
            for hand in hands:
                fingers = self.detector.fingersUp(hand)
                for idx, up in enumerate(fingers):
                    if up:
                        raised.add(idx)
        if raised:
            # choose the smallest finger index as the canonical label
            fidx = min(raised)
            label = FINGER_TO_LABEL.get(fidx, "NONE")
            conf = 0.9
        # Draw status text
        cv2.putText(img, f"cvzone: {sorted(list(raised))}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        return label, conf, img