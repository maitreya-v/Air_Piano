from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class Metrics:
    total: int = 0
    correct: int = 0
    avg_reaction_ms: float = 0.0

@dataclass
class AdaptiveCoach:
    metrics: Metrics = field(default_factory=Metrics)
    tempo_bpm: int = 60
    level: int = 1

    def update(self, correct: bool, reaction_ms: int):
        m = self.metrics
        m.total += 1
        if correct:
            m.correct += 1
        # moving average
        if m.total == 1:
            m.avg_reaction_ms = reaction_ms
        else:
            m.avg_reaction_ms = 0.8 * m.avg_reaction_ms + 0.2 * reaction_ms

        # difficulty rules
        acc = (m.correct / max(1, m.total))
        if acc > 0.9:
            self.tempo_bpm = min(140, self.tempo_bpm + 5)
        elif acc < 0.7:
            self.tempo_bpm = max(40, self.tempo_bpm - 5)

        if acc > 0.92 and self.level < 10:
            self.level += 1
        elif acc < 0.6 and self.level > 1:
            self.level -= 1

    def hint(self, last_pred: str, target: str) -> str:
        if last_pred == target:
            return "Great! Keep the posture steady."
        else:
            return "Focus the target finger lift and keep wrist centered."

    def summary(self) -> Dict[str, Any]:
        acc = (self.metrics.correct / max(1, self.metrics.total))
        return {
            "accuracy": round(acc, 3),
            "avg_reaction_ms": int(self.metrics.avg_reaction_ms),
            "tempo_bpm": self.tempo_bpm,
            "level": self.level
        }