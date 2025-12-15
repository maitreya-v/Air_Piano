from typing import Dict
from src.core.sound_engine import SoundEngine
from src.utils.mappings import LABEL_TO_NOTES

class FreePlayMode:
    def __init__(self, sound: SoundEngine):
        self.sound = sound

    def handle_prediction(self, label: str, confidence: float) -> Dict:
        notes = LABEL_TO_NOTES.get(label, [])
        if notes and confidence >= 0.5:
            self.sound.play_notes(notes, dur=0.3)
        return {"played": notes, "conf": round(confidence, 2)}