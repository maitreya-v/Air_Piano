from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

@dataclass
class GestureSample:
    x: np.ndarray  # feature vector
    y: int         # label id

class GestureClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.label_to_id: Dict[str, int] = {}
        self.id_to_label: Dict[int, str] = {}

    def fit(self, X: np.ndarray, y_labels: List[str]) -> None:
        # map labels to ids
        for lbl in y_labels:
            if lbl not in self.label_to_id:
                self.label_to_id[lbl] = len(self.label_to_id)
        y = np.array([self.label_to_id[lbl] for lbl in y_labels], dtype=np.int32)
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.model.fit(X, y)

    def predict_label(self, x: np.ndarray) -> Tuple[str, float]:
        proba = self.model.predict_proba(x[None, :])[0]
        idx = int(np.argmax(proba))
        return self.id_to_label[idx], float(proba[idx])

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_to_id': self.label_to_id
            }, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        self.model = obj['model']
        self.label_to_id = obj['label_to_id']
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}