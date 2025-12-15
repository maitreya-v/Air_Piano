from typing import Dict
import time
from src.core.feedback_engine import AdaptiveCoach
from src.core.sound_engine import SoundEngine
from src.utils.mappings import LABEL_TO_NOTES

class TutorialMode:
    """
    Step-by-step lesson player with one-shot gating:
      - Plays each target once when correctly detected
      - Advances to the next target
      - Requires the user to RELEASE the last gesture before accepting the next
      - Debounce to avoid re-triggering on consecutive frames
    """
    def __init__(
        self,
        coach: AdaptiveCoach,
        sound: SoundEngine,
        lesson=None,
        confidence_thresh: float = 0.6,
        debounce_ms: int = 500,
        release_frames: int = 5,
    ):
        # Default lesson if none provided
        self.lesson = lesson or ["NOTE_C4", "NOTE_D4", "NOTE_E4", "C_CHORD"]
        self.idx = 0
        self.coach = coach
        self.sound = sound

        # Gating params
        self.conf_thresh = confidence_thresh
        self.debounce_ms = debounce_ms
        self.release_frames_needed = release_frames

        # Internal state
        self._cooldown_until_ms = 0.0
        self._await_release = False
        self._lock_label = None
        self._release_count = 0

    def is_done(self) -> bool:
        return self.idx >= len(self.lesson)

    def target_label(self):
        return None if self.is_done() else self.lesson[self.idx]

    def _start_cooldown(self):
        self._cooldown_until_ms = time.time() * 1000 + self.debounce_ms

    def _in_cooldown(self) -> bool:
        return (time.time() * 1000) < self._cooldown_until_ms

    def handle_prediction(self, label: str, confidence: float, reaction_ms: int) -> Dict:
        target = self.target_label()

        # Finished lesson: return status, no further logic
        if target is None:
            return {
                "target": None,
                "pred": label,
                "conf": round(confidence, 2),
                "coach": self.coach.summary(),
                "done": True,
            }

        # If waiting for release of the last accepted label, require a few frames without that label
        if self._await_release:
            if label == self._lock_label and confidence >= 0.5:
                self._release_count = 0  # still holding; keep waiting
            else:
                self._release_count += 1
                if self._release_count >= self.release_frames_needed:
                    self._await_release = False
                    self._lock_label = None
                    self._release_count = 0
            # No coach update during release wait; just report current status
            return {
                "target": target,
                "pred": label,
                "conf": round(confidence, 2),
                "coach": self.coach.summary(),
                "done": False,
            }

        # Normal recognition path
        correct = (label == target and confidence >= self.conf_thresh)
        self.coach.update(correct, reaction_ms)

        if correct and not self._in_cooldown():
            # Play once
            notes = LABEL_TO_NOTES.get(target, [])
            if notes:
                self.sound.play_notes(notes, dur=0.3)

            # Advance to next index (can go past end so is_done() is True thereafter)
            self.idx += 1

            # Lock until user releases this gesture
            self._await_release = True
            self._lock_label = label
            self._start_cooldown()

        return {
            "target": self.target_label(),  # may be None after advancing
            "pred": label,
            "conf": round(confidence, 2),
            "coach": self.coach.summary(),
            "done": self.is_done(),
        }
