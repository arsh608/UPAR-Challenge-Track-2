"""Save OpenAI CLIP visual-tower weights as a plain state dict (run once, needs internet).

    pip install open_clip_torch
    python fetch_clip.py --arch ViT-B-16
"""
from __future__ import annotations

import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="ViT-B-16", choices=["ViT-B-16", "ViT-L-14"])
    ap.add_argument("--out-dir", default="weights")
    args = ap.parse_args()
    import open_clip

    model = open_clip.create_model(f"{args.arch}-quickgelu", pretrained="openai")
    sd = model.visual.state_dict()
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"clip_{args.arch}.pt")
    torch.save(sd, out)
    print(f"saved {len(sd)} tensors to {out}")


if __name__ == "__main__":
    main()
