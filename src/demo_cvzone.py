import time
import cv2

try:
    from cvzone.HandTrackingModule import HandDetector
except Exception as e:
    raise ImportError("cvzone is required for this demo. Install with `pip install cvzone`.") from e

from src.core.sound_engine import SoundEngine

# MIDI note numbers for D major scale chords
# Thumb-> D major (D F# A) ; Index-> E minor (E G B); Middle-> F# minor (F# A C#);
# Ring-> G major (G B D); Pinky-> A major (A C# E)
FINGER_TO_NOTES = {
    0: [62, 66, 69],  # Thumb -> D major   (D4=62, F#4=66, A4=69)
    1: [64, 67, 71],  # Index -> E minor   (E4, G4, B4)
    2: [66, 69, 73],  # Middle -> F# minor (F#4, A4, C#5)
    3: [67, 71, 74],  # Ring -> G major    (G4, B4, D5)
    4: [69, 73, 76],  # Pinky -> A major   (A4, C#5, E5)
}

def midi_nums_to_note_names(nums):
    # simple helper for our SoundEngine which expects note names if MIDI not available
    midi_to_name = {
        60:'C4', 61:'C#4', 62:'D4', 63:'D#4', 64:'E4', 65:'F4', 66:'F#4', 67:'G4', 68:'G#4', 69:'A4',
        70:'A#4', 71:'B4', 72:'C5', 73:'C#5', 74:'D5', 75:'D#5', 76:'E5'
    }
    return [midi_to_name.get(n, 'C4') for n in nums]

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam."); return

    detector = HandDetector(detectionCon=0.7, maxHands=2)
    sound = SoundEngine()
    sustain_time = 2.0  # seconds

    # Track which finger chord is currently sounding and when it started
    active = {}  # finger_index -> (start_time, notes)

    try:
        while True:
            ok, img = cap.read()
            if not ok: break
            img = cv2.flip(img, 1)

            hands, img = detector.findHands(img, draw=True)
            now = time.time()

            raised = set()
            if hands:
                for hand in hands:
                    fingers = detector.fingersUp(hand)  # [thumb..pinky] list of 0/1
                    for fi, up in enumerate(fingers):
                        if up: raised.add(fi)

            # Start chords for newly raised fingers
            for fi in raised:
                if fi not in active and fi in FINGER_TO_NOTES:
                    notes = midi_nums_to_note_names(FINGER_TO_NOTES[fi])
                    active[fi] = (now, notes)
                    sound.play_notes(notes, dur=0.25)  # quick attack; sustain managed below

            # Stop chords that have been released after sustain_time
            for fi in list(active.keys()):
                if fi not in raised and (now - active[fi][0]) >= sustain_time:
                    # send a short off by playing zero duration (SoundEngine uses note_off when using MIDI)
                    # Here we simply drop from active; SoundEngine stops after 'dur' anyway.
                    del active[fi]

            cv2.putText(img, f"Raised: {sorted(list(raised))}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.imshow("Guided Piano — cvzone demo (D major chords)", img)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        sound.close()

if __name__ == "__main__":
    main()