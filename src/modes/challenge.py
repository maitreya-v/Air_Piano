import random
from typing import Dict, List
from src.core.feedback_engine import AdaptiveCoach
from src.core.sound_engine import SoundEngine
from src.utils.mappings import LABEL_TO_NOTES

PATTERNS = [
    ["NOTE_C4", "NOTE_D4", "NOTE_E4"],
    ["G_CHORD", "C_CHORD"],
]

class ChallengeMode:
    def __init__(self, coach: AdaptiveCoach, sound: SoundEngine, patterns=None):
        self.coach = coach
        self.sound = sound
        self.pattern = random.choice(patterns or PATTERNS)
        self.pos = 0

    def reset(self):
        self.pattern = random.choice(patterns or PATTERNS)
        self.pos = 0

    def target_label(self) -> str:
        return self.pattern[self.pos]

    def handle_prediction(self, label: str, confidence: float, reaction_ms: int) -> Dict:
        target = self.target_label()
        correct = (label == target and confidence >= 0.6)
        self.coach.update(correct, reaction_ms)
        if correct:
            self.sound.play_notes(LABEL_TO_NOTES[target], dur=0.25)
            self.pos += 1
            if self.pos >= len(self.pattern):
                self.reset()
        return {
            "target": target,
            "pred": label,
            "conf": round(confidence, 2),
            "coach": self.coach.summary()
        }