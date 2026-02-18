from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

def features_from_pair(slide_vec: np.ndarray, seg_vec: np.ndarray) -> np.ndarray:
    """
    Build feature vector for MLP:
      [cos_sim, |a-b|, a*b, a, b]
    where a,b are embeddings (already normalized).
    """
    a = slide_vec.astype(np.float32)
    b = seg_vec.astype(np.float32)

    cos = float(np.dot(a, b))  # since normalized
    absdiff = np.abs(a - b)
    prod = a * b
    feat = np.concatenate([[cos], absdiff, prod, a, b], axis=0)
    return feat.astype(np.float32)

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
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

@dataclass
class ConfidenceModel:
    model: ConfidenceMLP
    device: str = "cpu"

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(X).to(self.device)
            logits = self.model(t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

def save_confidence_model(cm: ConfidenceModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": cm.model.state_dict()}, path)

def load_confidence_model(path: str | Path, in_dim: int, device: str = "cpu") -> ConfidenceModel:
    ckpt = torch.load(str(path), map_location=device)
    m = ConfidenceMLP(in_dim=in_dim)
    m.load_state_dict(ckpt["state_dict"])
    m.to(device)
    return ConfidenceModel(model=m, device=device)
