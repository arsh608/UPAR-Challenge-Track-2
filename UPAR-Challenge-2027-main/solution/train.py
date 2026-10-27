"""Fine-tune CLIP ViT for the 40 UPAR attributes.

    # development run: train on train, validate on val every epoch
    python train.py --cache cache --train-splits train --val-split val --out runs/dev
    # final run: train on train+val (allowed by the rules), no validation
    python train.py --cache cache --train-splits train val --out runs/final
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import CacheDataset, gpu_augment, read_queries
from evaluate import predict_dataset
from retrieval import par_metrics, retrieval_metrics, score_matrix
from upar_model import PARModel, load_clip_visual, param_groups, save_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--train-splits", nargs="+", default=["train"])
    ap.add_argument("--val-split", default=None)
    ap.add_argument("--queries", default="../data/annotations/task2/val/queries.csv")
    ap.add_argument("--arch", default="ViT-B-16")
    ap.add_argument("--clip-weights", default="weights/clip_ViT-B-16.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=6e-5, help="lr of the last CLIP block")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--layer-decay", type=float, default=0.75)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=0.5, help="epochs")
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--loss", choices=["bce", "wbce"], default="bce")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop early")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/dev")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    use_amp = dev.type == "cuda"

    train_ds = CacheDataset(args.cache, args.train_splits)
    loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True,
                        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    print(f"train images: {len(train_ds)}  steps/epoch: {len(loader)}  device: {dev}  amp: {amp_dtype}")

    model = PARModel(args.arch)
    if args.clip_weights and os.path.exists(args.clip_weights):
        load_clip_visual(model, torch.load(args.clip_weights, map_location="cpu", weights_only=True))
        print("loaded CLIP weights", args.clip_weights)
    else:
        print("WARNING: no CLIP weights found, training from scratch")
    model.backbone.transformer.grad_ckpt = args.grad_ckpt
    model.to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(param_groups(model, args.lr, args.head_lr, args.wd, args.layer_decay), betas=(0.9, 0.999))
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    total = args.epochs * len(loader)
    warm = int(args.warmup * len(loader))

    pos = train_ds.labels.mean(0).clamp(1e-4, 1 - 1e-4).to(dev)
    w_pos, w_neg = torch.exp(1 - pos), torch.exp(pos)  # "strong baseline" weighted BCE

    val = None
    if args.val_split:
        val = CacheDataset(args.cache, [args.val_split])
        val_q = torch.from_numpy(read_queries(args.queries))

    state_path = os.path.join(args.out, "last_state.pt")
    start_epoch, step, best = 0, 0, -1.0
    if os.path.exists(state_path):  # resume after a Colab disconnect
        st = torch.load(state_path, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        if st["scaler"] and scaler.is_enabled():
            scaler.load_state_dict(st["scaler"])
        start_epoch, step, best = st["epoch"] + 1, st["step"], st["best"]
        print(f"resumed from epoch {start_epoch}")

    log = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, run_loss = time.time(), 0.0
        for i, (x, y) in enumerate(loader):
            lr_f = step / max(1, warm) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_f
            x = gpu_augment(x.to(dev, non_blocking=True))
            y = y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
            logits = logits.float()
            if args.loss == "wbce":
                w = y * w_pos + (1 - y) * w_neg
                loss = F.binary_cross_entropy_with_logits(logits, y, weight=w)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            d = min(args.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.state_dict().values(), model.state_dict().values()):
                    if pe.is_floating_point():
                        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            step += 1
            run_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"ep {epoch} it {i + 1}/{len(loader)} loss {run_loss / (i + 1):.4f} "
                      f"{(i + 1) * args.bs / (time.time() - t0):.0f} img/s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        entry = {"epoch": epoch, "loss": run_loss / max(1, i + 1), "time_min": (time.time() - t0) / 60}

        save_checkpoint(ema, os.path.join(args.out, "model_ema.pt"))
        if val is not None:
            probs = predict_dataset(ema, val, dev, bs=256, workers=args.workers)
            entry.update({f"val_{k}": v for k, v in par_metrics(probs, val.labels).items()})
            sim = score_matrix(probs.to(dev), val_q.to(dev))
            entry.update({f"val_{k}": v for k, v in retrieval_metrics(sim, val_q, val.labels).items()})
            if entry["val_mADM"] > best:
                best = entry["val_mADM"]
                save_checkpoint(ema, os.path.join(args.out, "model_best.pt"))
        log.append(entry)
        print(json.dumps(entry), flush=True)
        with open(os.path.join(args.out, "log.jsonl"), "a") as fh:
            fh.write(json.dumps(entry) + "\n")
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "step": step, "best": best}, state_path)
        if args.max_steps and step >= args.max_steps:
            break
    os.remove(state_path)  # training finished; the large optimizer state is no longer needed
    print("done. best val mADM:", best)


if __name__ == "__main__":
    main()
