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
from app.confidence import ConfidenceMLP, ConfidenceModel, features_from_pair, save_confidence_model

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--youtube", type=str)
    src.add_argument("--audio", type=str)
    ap.add_argument("--slides", type=str, required=True)
    ap.add_argument("--whisper-model", type=str, default="base", choices=["tiny","base","small","medium"])
    ap.add_argument("--min-cos-pos", type=float, default=0.35)
    ap.add_argument("--neg-per-pos", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--out", type=str, default="runs/models/confidence_mlp.pt")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    # 1) transcript
    if args.youtube:
        audio_path = download_audio(args.youtube, out_dir="runs/audio")
    else:
        audio_path = Path(args.audio)
    tr = transcribe_whisper(audio_path, model_size=args.whisper_model)
    segments = [s for s in tr["segments"] if s["text"].strip()]
    if not segments:
        raise SystemExit("No transcript segments. Try a clearer audio or a bigger whisper model.")

    # 2) slides
    slides = load_slides(args.slides, use_ocr_if_empty=False)
    slide_payload = [{"slide_id": s.slide_id, "text": s.text} for s in slides]

    slide_texts = [s["text"] if s["text"] else "(empty)" for s in slide_payload]
    seg_texts = [s["text"] for s in segments]

    S = embed_texts(slide_texts)  # normalized
    T = embed_texts(seg_texts)

    # cosine sims = dot product because normalized
    sims = S @ T.T
    best_idx = sims.argmax(axis=1)
    best_score = sims.max(axis=1)

    X_list, y_list = [], []

    rng = random.Random(42)
    for i in range(len(slide_payload)):
        j = int(best_idx[i])
        sc = float(best_score[i])
        if sc < args.min_cos_pos:
            continue  # skip weak positives

        # positive
        feat = features_from_pair(S[i], T[j])
        X_list.append(feat)
        y_list.append(1.0)

        # negatives
        cand = list(range(len(segments)))
        cand.remove(j)
        rng.shuffle(cand)
        for k in cand[: args.neg_per_pos]:
            featn = features_from_pair(S[i], T[k])
            X_list.append(featn)
            y_list.append(0.0)

    if len(y_list) < 20:
        raise SystemExit("Not enough training pairs. Use longer audio or lower --min-cos-pos (e.g. 0.25).")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    in_dim = X.shape[1]
    model = ConfidenceMLP(in_dim=in_dim).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for ep in range(1, args.epochs + 1):
        total = 0.0
        for xb, yb in dl:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch {ep}/{args.epochs} loss={total/len(dl):.4f}")

    cm = ConfidenceModel(model=model, device=args.device)
    save_confidence_model(cm, args.out)
    print(f"✅ saved confidence model: {args.out}\n(in_dim={in_dim})")

if __name__ == "__main__":
    main()
