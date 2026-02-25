from __future__ import annotations
import argparse
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.audio import download_audio, transcribe_whisper
from app.slides import load_slides
from app.align import embed_texts
from app.confidence import (
    ConfidenceMLP,
    ConfidenceModel,
    features_from_pair,
    save_confidence_model,
)

def main():
    ap = argparse.ArgumentParser()

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--youtube", type=str)
    src.add_argument("--audio", type=str)

    ap.add_argument("--slides", type=str, required=True)
    ap.add_argument("--whisper-model", type=str, default="base")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--out", type=str, default="runs/models/confidence_mlp.pt")

    args = ap.parse_args()

    # ---------------------------
    # 1. TRANSCRIPT
    # ---------------------------

    if args.youtube:
        audio_path = download_audio(args.youtube)
    else:
        audio_path = Path(args.audio)

    tr = transcribe_whisper(audio_path, model_size=args.whisper_model)

    segments = [s for s in tr["segments"] if s["text"].strip()]
    if not segments:
        raise SystemExit("❌ No transcript segments")

    # ---------------------------
    # 2. SLIDES
    # ---------------------------

    slides = load_slides(args.slides)
    slide_texts = [s.text for s in slides]
    seg_texts = [s["text"] for s in segments]

    # ---------------------------
    # 3. EMBEDDINGS
    # ---------------------------

    S = embed_texts(slide_texts)
    T = embed_texts(seg_texts)

    sims = S @ T.T

    X_list, y_list = []
    rng = random.Random(42)

    # ---------------------------
    # 4. BUILD TRAIN DATA
    # ---------------------------

    for i in range(len(slides)):
        sim_row = sims[i]

        # TOP 3 positives
        top_pos = sim_row.argsort()[-3:]

        for j in top_pos:
            if sim_row[j] < 0.3:
                continue

            X_list.append(features_from_pair(S[i], T[j]))
            y_list.append(1.0)

        # HARD NEGATIVES (high similarity but wrong)
        neg_candidates = sim_row.argsort()[-10:]
        rng.shuffle(neg_candidates)

        for j in neg_candidates[:5]:
            X_list.append(features_from_pair(S[i], T[j]))
            y_list.append(0.0)

    if len(y_list) < 50:
        raise SystemExit("❌ Not enough training data")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)

    # ---------------------------
    # 5. TRAIN MODEL
    # ---------------------------

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    model = ConfidenceMLP(in_dim=X.shape[1]).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()

    for ep in range(1, args.epochs + 1):
        total = 0.0

        for xb, yb in dl:
            xb, yb = xb.to(args.device), yb.to(args.device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)

            loss.backward()
            opt.step()

            total += loss.item()

        print(f"epoch {ep} loss={total/len(dl):.4f}")

    # ---------------------------
    # 6. SAVE MODEL
    # ---------------------------

    cm = ConfidenceModel(model=model, device=args.device)
    save_confidence_model(cm, args.out)

    print(f"✅ saved model → {args.out}")


if __name__ == "__main__":
    main()