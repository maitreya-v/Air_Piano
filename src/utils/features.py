from typing import Dict, Any
import numpy as np

# MediaPipe index pairs to compute distances/angles
FINGERTIP_IDX = [4, 8, 12, 16, 20]
WRIST_IDX = 0

def _pairwise_dists(points: np.ndarray) -> np.ndarray:
    # points: (21, 2) normalized [0,1]
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)  # (21, 21)
    return dists

def extract_features(landmarks: np.ndarray) -> Dict[str, Any]:
    """Turn 21x2 landmarks into a compact, translation/scale-insensitive feature vector."""
    # Normalize by wrist-centered coordinates and scale by palm size
    wrist = landmarks[WRIST_IDX]
    rel = landmarks - wrist
    scale = np.linalg.norm(rel[9]) + 1e-6  # use index MCP as a scale proxy
    rel /= scale

    # Distances between fingertips and wrist
    fingertip_d = [np.linalg.norm(rel[idx]) for idx in FINGERTIP_IDX]

    # Pairwise distances of fingertips (upper-triangular)
    d = _pairwise_dists(rel[FINGERTIP_IDX])
    tri = d[np.triu_indices(len(FINGERTIP_IDX), k=1)].tolist()

    # Simple angles for two fingers vs wrist line (approx)
    angles = []
    for idx in FINGERTIP_IDX:
        v = rel[idx]
        ang = float(np.arctan2(v[1], v[0]))
        angles.append(ang)

    vec = fingertip_d + tri + angles
    return {"vector": np.asarray(vec, dtype=np.float32)}