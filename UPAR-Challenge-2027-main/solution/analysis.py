"""Report figures, the scoring ablation and the domain-shift robustness check.

Runs on CPU from the saved validation probabilities (no images or GPU needed):

    python analysis.py --runs ../../runs --out ../../report/figures
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from retrieval import retrieval_metrics, score_matrix  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})


def read_log(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def training_curves(runs, out):
    dev = read_log(os.path.join(runs, "dev", "log.jsonl"))
    ep = [d["epoch"] + 1 for d in dev]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    ax[0].plot(ep, [d["loss"] for d in dev], color=BLUE, lw=2, marker="o", ms=4, label="train, 10 ep")
    for name, color, marker, label in [("final", ORANGE, "s", "train+val, 10 ep"),
                                       ("final3", AQUA, "^", "train+val, 3 ep (submitted)")]:
        path = os.path.join(runs, name, "log.jsonl")
        if os.path.exists(path):
            log = read_log(path)
            ax[0].plot([d["epoch"] + 1 for d in log], [d["loss"] for d in log], color=color, lw=2,
                       marker=marker, ms=4, label=label)
    ax[0].set_title("Training loss (BCE)", fontsize=9, color=INK)
    ax[0].legend(frameon=False, fontsize=6.5)
    best = int(np.argmax([d["val_mADM"] for d in dev]))
    for a, key, title in [(ax[1], "val_mADM", "Validation mADM"), (ax[2], "val_mAP", "Validation mAP")]:
        v = [d[key] for d in dev]
        a.plot(ep, v, color=BLUE, lw=2, marker="o", ms=4)
        a.plot(ep[best], v[best], "o", ms=9, mfc="none", mec=INK, mew=1.2)
        a.text(0.97, 0.9, f"best: epoch {ep[best]}\n({v[best]:.3f})", transform=a.transAxes, ha="right",
               va="top", fontsize=7.5, color=INK)
        a.set_title(title, fontsize=9, color=INK)
    for a in ax:
        a.set_xlabel("epoch")
        a.set_xticks(range(1, 11))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "training_curves.png"), dpi=200)
    plt.close(fig)


def ablation(probs, queries, labels, scoring, runs, out):
    rows = []
    for name, p, cfg in [("Organizer baseline (attribute prior + L1)", labels.mean(0).expand_as(probs), {"ll": 0, "alpha": 1}),
                         ("CLIP model + L1 distance", probs, {"ll": 0, "alpha": 1}),
                         ("+ log-likelihood scoring", probs, {"ll": 1, "eps": 1e-4}),
                         ("+ agreement term + query normalisation (ours)", probs, scoring)]:
        m = retrieval_metrics(score_matrix(p, queries, cfg), queries, labels)
        rows.append((name, m))
        print(f"{name:48s} mADM {m['mADM']:.4f}  mAP {m['mAP']:.4f}  R-1 {m['Rank-1']:.4f}")
    with open(os.path.join(runs, "ablation.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    fig, ax = plt.subplots(figsize=(5.2, 2.0))
    y = np.arange(len(rows))[::-1]
    v = [m["mADM"] for _, m in rows]
    ax.barh(y, v, height=0.55, color=[GRID] + [BLUE] * (len(rows) - 1), edgecolor="white", linewidth=2)
    for yi, vi in zip(y, v):
        ax.text(vi + 0.01, yi, f"{vi:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace(" (", "\n(").replace(" + query", "\n+ query") for n, _ in rows], fontsize=7.5)
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("validation mADM")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ablation.png"), dpi=200)
    plt.close(fig)


def per_attribute(probs, labels, names, out, runs):
    pred = (probs > 0.5).float()
    tp, tn = (pred * labels).sum(0), ((1 - pred) * (1 - labels)).sum(0)
    ba = ((tp / labels.sum(0).clamp_min(1) + tn / (1 - labels).sum(0).clamp_min(1)) / 2).numpy()
    pos = labels.mean(0).numpy()
    order = np.argsort(ba)
    with open(os.path.join(runs, "attribute_accuracy.json"), "w") as fh:
        json.dump([{"attribute": names[i], "balanced_accuracy": float(ba[i]), "positive_rate": float(pos[i])}
                   for i in order], fh, indent=1)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.bar(range(40), ba[order], width=0.7, color=[ORANGE if pos[i] < 0.02 else BLUE for i in order],
           edgecolor="white", linewidth=1)
    ax.set_xticks(range(40))
    ax.set_xticklabels([names[i] for i in order], rotation=90, fontsize=6)
    ax.set_ylim(0.5, 1)
    ax.set_ylabel("balanced accuracy")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color=ORANGE, label="rare (<2% positives)"), Patch(color=BLUE, label="other attributes")],
              frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "per_attribute.png"), dpi=200)
    plt.close(fig)


def robustness(probs, queries, labels, scoring, runs, n_queries=1200, seed=0):
    """Does our scoring stay ahead of L1 when the probabilities are degraded like under domain shift?"""
    torch.manual_seed(seed)
    qs = queries[torch.randperm(len(queries))[:n_queries]]
    logit = torch.logit(probs.clamp(1e-6, 1 - 1e-6))
    shifts = {"clean": logit,
              "noise (std 1)": logit + torch.randn_like(logit),
              "noise (std 2) + over-confident": 1.5 * logit + 2 * torch.randn_like(logit),
              "attribute prior shift": logit + torch.randn(40)}
    cfgs = {"L1": {"ll": 0, "alpha": 1}, "log-likelihood": {"eps": 1e-3}, "ours": scoring}
    print("\nmADM under simulated shift:", "  ".join(f"{k:>14s}" for k in cfgs))
    table = {}
    for name, lg in shifts.items():
        p = torch.sigmoid(lg)
        vals = [retrieval_metrics(score_matrix(p, qs, c), qs, labels)["mADM"] for c in cfgs.values()]
        table[name] = dict(zip(cfgs, vals))
        print(f"  {name:32s}", "  ".join(f"{v:14.4f}" for v in vals))
    with open(os.path.join(runs, "robustness.json"), "w") as fh:
        json.dump(table, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="../../runs")
    ap.add_argument("--ann", default="../data/annotations")
    ap.add_argument("--out", default="../../report/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.ann, "val.csv")) as fh:
        names = fh.readline().strip("# \n").split(",")[1:]
    labels = torch.tensor(np.loadtxt(os.path.join(args.ann, "val.csv"), delimiter=",", skiprows=1,
                                     usecols=range(1, 41)), dtype=torch.float32)
    queries = torch.tensor(np.loadtxt(os.path.join(args.ann, "task2", "val", "queries.csv"), delimiter=",",
                                      skiprows=1), dtype=torch.float32)
    probs = torch.from_numpy(np.load(os.path.join(args.runs, "dev", "val_probs.npy"))).float()
    with open(os.path.join(args.runs, "dev", "scoring.json")) as fh:
        scoring = json.load(fh)
    training_curves(args.runs, args.out)
    ablation(probs, queries, labels, scoring, args.runs, args.out)
    per_attribute(probs, labels, names, args.out, args.runs)
    robustness(probs, queries, labels, scoring, args.runs)


if __name__ == "__main__":
    main()
