from typing import List, Optional
import time

# Try MIDI via pygame; otherwise fall back to simpleaudio WAV playback.
try:
    import pygame.midi as midi
    _MIDI_AVAILABLE = True
except Exception:
    _MIDI_AVAILABLE = False

try:
    import simpleaudio as sa
    _SA_AVAILABLE = True
except Exception:
    _SA_AVAILABLE = False

NOTE_TO_MIDI = {
    # Basic octave mapping
    'C4': 60, 'D4': 62, 'E4': 64, 'F4': 65, 'G4': 67, 'A4': 69, 'B4': 71,
    'C5': 72, 'D5': 74, 'E5': 76,
}

class SoundEngine:
    def __init__(self, wav_folder: Optional[str] = None):
        self.wav_folder = wav_folder
        self.midi_out = None
        if _MIDI_AVAILABLE:
            midi.init()
            try:
                self.midi_out = midi.Output(0)
            except Exception:
                self.midi_out = None

    def play_notes(self, notes: List[str], dur: float = 0.5, velocity: int = 90):
        if self.midi_out is not None:
            # MIDI chord
            for n in notes:
                midi_num = NOTE_TO_MIDI.get(n, 60)
                self.midi_out.note_on(midi_num, velocity)
            time.sleep(dur)
            for n in notes:
                midi_num = NOTE_TO_MIDI.get(n, 60)
                self.midi_out.note_off(midi_num, velocity)
        elif _SA_AVAILABLE and self.wav_folder:
            # very simple sequential WAV playback
            for n in notes:
                try:
                    wave = sa.WaveObject.from_wave_file(f"{self.wav_folder}/{n}.wav")
                    play = wave.play()
                    play.wait_done()
                except Exception:
                    pass
        else:
            # Fallback: no audio backend — simulate with print
            print(f"[SOUND] {notes} for {dur:.2f}s" )

    def close(self):
        if self.midi_out is not None:
            self.midi_out.close()
            try:
                midi.quit()
            except Exception:
                pass