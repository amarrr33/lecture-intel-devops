from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------
# Feature Engineering
# ---------------------------

def features_from_pair(slide_vec: np.ndarray, seg_vec: np.ndarray) -> np.ndarray:
    """
    Build feature vector for MLP:
      [cos_sim, |a-b|, a*b, a, b]

    Assumes embeddings are normalized.
    """
    a = slide_vec.astype(np.float32)
    b = seg_vec.astype(np.float32)

    # Cosine similarity (safe)
    cos = float(np.clip(np.dot(a, b), -1.0, 1.0))

    absdiff = np.abs(a - b)
    prod = a * b

    feat = np.concatenate(
        [[cos], absdiff, prod, a, b],
        axis=0
    )

    return feat.astype(np.float32)


# ---------------------------
# Model
# ---------------------------

class ConfidenceMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------
# Wrapper
# ---------------------------

@dataclass
class ConfidenceModel:
    model: ConfidenceMLP
    device: str = "cpu"

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probability scores (0–1)
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float32)

        self.model.eval()

        with torch.no_grad():
            t = torch.from_numpy(X.astype(np.float32)).to(self.device)

            logits = self.model(t)

            probs = torch.sigmoid(logits)

            return probs.cpu().numpy()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Binary predictions based on threshold
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.float32)


# ---------------------------
# Save / Load
# ---------------------------

def save_confidence_model(cm: ConfidenceModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "state_dict": cm.model.state_dict(),
        "in_dim": next(cm.model.parameters()).shape[1] if cm.model else None
    }, path)


def load_confidence_model(
    path: str | Path,
    in_dim: int,
    device: str = "cpu"
) -> ConfidenceModel:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Confidence model not found: {path}")

    ckpt = torch.load(str(path), map_location=device)

    model = ConfidenceMLP(in_dim=in_dim)

    # Safe load
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)

    return ConfidenceModel(model=model, device=device)