"""UPAR 2027 Track 2 submission: fine-tuned CLIP ViT attribute model + likelihood ranking.

1. predict 40 attribute probabilities for every gallery image (flip TTA, model ensemble)
2. score every query against every image by the log-likelihood of an exact match plus the
   expected number of agreeing attributes, then normalise over queries (each gallery image
   belongs to exactly one query).
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from retrieval import score_matrix  # noqa: E402
from upar_model import load_checkpoint  # noqa: E402

_MODELS: list = []
_SCORING: dict = {}
_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 256


def load_model() -> None:
    global _SCORING
    if _MODELS:
        return
    for path in sorted(glob.glob(os.path.join(HERE, "assets", "model*.pt"))):
        m = load_checkpoint(path).to(_DEV).eval()
        if _DEV.type == "cuda":
            m.half()
        _MODELS.append(m)
        print(f"[run.py] loaded {os.path.basename(path)} ({m.arch}, {m.img_size})", flush=True)
    with open(os.path.join(HERE, "assets", "scoring.json"), "r", encoding="utf-8") as fh:
        _SCORING = json.load(fh)
    print(f"[run.py] device {_DEV}, scoring {_SCORING}", flush=True)


def _load(path: str, size) -> np.ndarray:
    try:
        img = Image.open(path).convert("RGB").resize((size[1], size[0]), Image.BILINEAR)
        return np.asarray(img)
    except Exception as exc:  # a broken file must not kill the whole run
        print(f"[run.py] could not read {path}: {exc}", flush=True)
        return np.zeros((size[0], size[1], 3), dtype=np.uint8)


@torch.no_grad()
def predict_attributes(gallery: list[dict[str, Any]]) -> torch.Tensor:
    paths = [g["image_path"] for g in gallery]
    sizes = sorted({m.img_size for m in _MODELS})
    probs = torch.zeros(len(paths), 40, device=_DEV)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4))) as pool:
        for size in sizes:
            models = [m for m in _MODELS if m.img_size == size]
            batches = [paths[i:i + BATCH] for i in range(0, len(paths), BATCH)]
            pending = pool.map(lambda p: _load(p, size), batches[0]) if batches else None
            for b, batch in enumerate(batches):
                imgs = np.stack(list(pending))
                if b + 1 < len(batches):  # decode the next batch while the GPU works
                    pending = pool.map(lambda p: _load(p, size), batches[b + 1])
                x = torch.from_numpy(imgs).to(_DEV).permute(0, 3, 1, 2).float() / 255.0
                if _DEV.type == "cuda":
                    x = x.half()
                s = b * BATCH
                for m in models:
                    p = torch.sigmoid(m(x).float()) + torch.sigmoid(m(x.flip(-1)).float())
                    probs[s:s + len(batch)] += p / (2 * len(_MODELS))
                if b % 20 == 0:
                    print(f"[run.py] {s + len(batch)}/{len(paths)} images, {time.time() - t0:.0f}s", flush=True)
    return probs


def rank_gallery(sample: dict[str, Any]) -> dict[str, Any]:
    load_model()
    t0 = time.time()
    probs = predict_attributes(sample["gallery"])
    queries = torch.as_tensor(np.asarray(sample["queries"], dtype=np.float32), device=_DEV)
    sim = score_matrix(probs, queries, _SCORING)
    sim = torch.nan_to_num(sim, nan=-1e9, posinf=1e9, neginf=-1e9)
    print(f"[run.py] {queries.shape[0]} queries x {probs.shape[0]} images in {time.time() - t0:.0f}s", flush=True)
    return {"similarities": sim.cpu().numpy()}
