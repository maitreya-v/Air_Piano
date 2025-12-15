import cv2, time
import numpy as np
from src.core.hand_tracking import HandTracker
from src.utils.features import extract_features

# Optional: a stub classifier that returns a constant label
class _StubClassifier:
    def predict_label(self, x):
        return ("NOTE_C4", 0.7)

def main():
    tracker = HandTracker(max_hands=1)
    clf = _StubClassifier()  # replace with a real loaded model
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    last_ts = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hands = tracker.process(frame)
        frame_drawn = tracker.draw(frame.copy(), hands)

        if hands:
            feats = extract_features(hands[0].points)["vector"]
            label, conf = clf.predict_label(feats)
            dt_ms = int((time.time() - last_ts) * 1000)
            last_ts = time.time()
            cv2.putText(frame_drawn, f"Pred: {label} ({conf:.2f})  dt={dt_ms}ms", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("Guided Piano — Demo", frame_drawn)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()