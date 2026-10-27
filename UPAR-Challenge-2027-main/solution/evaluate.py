"""Evaluate a checkpoint on the validation split and tune the retrieval scoring.

    python evaluate.py --ckpt runs/dev/model_best.pt --cache cache --split val --out runs/dev

Writes scoring.json (best scoring hyper-parameters, shipped in the submission),
results.json (all metrics, for the report) and val_probs.npy.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from retrieval import DEFAULT_SCORING, par_metrics, retrieval_metrics, score_matrix


@torch.no_grad()
def predict_dataset(model, ds, dev, bs=256, workers=2, tta=True) -> torch.Tensor:
    """Attribute probabilities (N x 40) on CPU, averaged over the image and its mirror."""
    model.eval()
    amp_dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    out = []
    for x, _ in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers):
        x = x.to(dev, non_blocking=True).float() / 255.0
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=dev.type == "cuda"):
            p = torch.sigmoid(model(x).float())
            if tta:
                p = (p + torch.sigmoid(model(x.flip(-1)).float())) / 2
        out.append(p.float().cpu())
    return torch.cat(out)


def tune_scoring(probs, queries, labels, dev):
    """Small staged grid search on the validation queries; mADM is the ranking metric."""
    probs, queries = probs.to(dev), queries.to(dev)
    results = []

    def run(cfg):
        m = retrieval_metrics(score_matrix(probs, queries, cfg), queries, labels)
        results.append({**cfg, **m})
        print(f"  {cfg} -> mADM {m['mADM']:.4f} mAP {m['mAP']:.4f} R1 {m['Rank-1']:.4f}", flush=True)
        return m["mADM"]

    # baseline-style L1 ranking for reference (alpha huge = expected agreement dominates)
    run({**DEFAULT_SCORING, "ll": 0.0, "alpha": 1.0})
    best = dict(DEFAULT_SCORING)
    best_score = -1
    for eps, alpha in itertools.product([1e-4, 1e-3, 1e-2, 3e-2], [0.0, 1.0, 3.0, 10.0]):
        cfg = {**best, "eps": eps, "alpha": alpha, "beta": 0.0}
        s = run(cfg)
        if s > best_score:
            best_score, best_stage1 = s, cfg
    best = best_stage1
    for beta, tau in itertools.product([0.5, 1.0, 2.0, 4.0], [1.0, 3.0, 10.0]):
        cfg = {**best_stage1, "beta": beta, "tau": tau}
        s = run(cfg)
        if s > best_score:
            best_score, best = s, cfg
    return best, results


def main():
    from data import CacheDataset, read_queries
    from upar_model import load_checkpoint

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--split", default="val")
    ap.add_argument("--queries", default="../data/annotations/task2/val/queries.csv")
    ap.add_argument("--probs", nargs="*", default=None, help="average saved val_probs.npy files (ensemble) instead of --ckpt")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--out", default="runs/dev")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = CacheDataset(args.cache, [args.split])
    if args.probs:
        probs = torch.from_numpy(np.mean([np.load(p) for p in args.probs], axis=0))
    else:
        model = load_checkpoint(args.ckpt).to(dev)
        probs = predict_dataset(model, ds, dev, workers=args.workers)
        np.save(os.path.join(args.out, "val_probs.npy"), probs.numpy())
    queries = torch.from_numpy(read_queries(args.queries))

    res = {"track1": par_metrics(probs, ds.labels)}
    print("Track 1 metrics:", res["track1"])
    best, grid = tune_scoring(probs, queries, ds.labels, dev)
    res["track2_default"] = grid[1]
    res["track2_l1_baseline_scoring"] = grid[0]
    res["track2_best"] = next(r for r in grid if all(r[k] == best[k] for k in best))
    res["grid"] = grid

    # per source dataset (queries restricted to the ones present in that sub-gallery)
    res["per_dataset"] = {}
    names = np.array([p.split("/")[0] for p in ds.paths])
    bits = torch.tensor([1 << k for k in range(40)], dtype=torch.int64)
    q_code = (queries.long() * bits).sum(1).numpy()
    for name in np.unique(names):
        idx = torch.from_numpy(np.nonzero(names == name)[0])
        sub_lab = ds.labels[idx]
        present = np.isin(q_code, (sub_lab.long() * bits).sum(1).numpy())
        q = queries[torch.from_numpy(present)]
        sim = score_matrix(probs[idx].to(dev), q.to(dev), best)
        res["per_dataset"][str(name)] = {"n_images": len(idx), "n_queries": int(q.shape[0]),
                                         **retrieval_metrics(sim, q, sub_lab)}
    print("best scoring:", best)
    print(json.dumps(res["track2_best"], indent=1))
    print(json.dumps(res["per_dataset"], indent=1))
    with open(os.path.join(args.out, "scoring.json"), "w") as fh:
        json.dump(best, fh, indent=2)
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
