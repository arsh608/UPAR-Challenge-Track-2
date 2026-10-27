"""Query-to-gallery scoring and the Track 2 metrics (mADM, mAP, Rank-k, mINP).

Shared by evaluate.py and the submission's run.py. Torch only.
"""
from __future__ import annotations

import torch

DEFAULT_SCORING = {"ll": 1.0, "eps": 1e-3, "alpha": 0.0, "beta": 0.0, "tau": 1.0}


@torch.no_grad()
def score_matrix(probs: torch.Tensor, queries: torch.Tensor, cfg: dict | None = None,
                 chunk: int = 8192) -> torch.Tensor:
    """(Q x G) similarities, larger = better match.

    s = log P(image has exactly the query's attributes)     (independent Bernoullis)
      + alpha * expected number of agreeing attributes       (credits near misses, i.e. mADM)
    then, with beta > 0, query-side normalisation: every gallery image matches exactly one
    query, so an image that fits many queries equally well is pushed down
      s += beta * log_softmax_over_queries(s / tau).
    """
    cfg = {**DEFAULT_SCORING, **(cfg or {})}
    q = queries.float()
    out = torch.empty(q.shape[0], probs.shape[0], device=q.device)
    for s in range(0, probs.shape[0], chunk):
        p = probs[s:s + chunk].float().to(q.device)
        sim = torch.zeros(q.shape[0], p.shape[0], device=q.device)
        if cfg["ll"]:
            pc = p.clamp(cfg["eps"], 1 - cfg["eps"])
            sim += cfg["ll"] * (q @ pc.log().T + (1 - q) @ (1 - pc).log().T)
        if cfg["alpha"]:
            sim += cfg["alpha"] * (q @ p.T + (1 - q) @ (1 - p).T)
        if cfg["beta"]:
            sim += cfg["beta"] * torch.log_softmax(sim / cfg["tau"], dim=0)
        out[:, s:s + chunk] = sim
    return out


@torch.no_grad()
def retrieval_metrics(sim: torch.Tensor, queries: torch.Tensor, gallery_attrs: torch.Tensor,
                      chunk: int = 256) -> dict:
    """Reimplementation of the Codabench Track 2 metrics. A gallery image is a correct
    match when all 40 attributes equal the query. Ties keep gallery order (stable sort)."""
    dev = sim.device
    q_all = queries.float().to(dev)
    ga = gallery_attrs.float().to(dev)
    n_attr = ga.shape[1]
    G = ga.shape[0]
    ranks = torch.arange(1, G + 1, device=dev, dtype=torch.float32)
    tot = {"mADM": 0.0, "mAP": 0.0, "Rank-1": 0.0, "Rank-5": 0.0, "Rank-10": 0.0, "mINP": 0.0}
    n_q = 0
    for s in range(0, q_all.shape[0], chunk):
        q = q_all[s:s + chunk]
        dom = (q @ ga.T + (1 - q) @ (1 - ga).T) / n_attr  # share of agreeing attributes
        correct = dom > 1 - 1e-6
        mean, mx = dom.mean(1, keepdim=True), dom.max(1, keepdim=True).values
        ndom = ((dom - mean) / (mx - mean).clamp_min(1e-12)).clamp_min(0)
        order = torch.sort(sim[s:s + chunk], dim=1, descending=True, stable=True).indices
        c = correct.gather(1, order).float()
        nd = ndom.gather(1, order)
        R = c.sum(1)
        valid = R > 0
        hits = c.cumsum(1)
        ap = (hits / ranks * c).sum(1) / R.clamp_min(1)
        adm = (nd.cumsum(1) / ranks * c).sum(1) / R.clamp_min(1)
        first = torch.where(c > 0, ranks, torch.full_like(ranks, G + 1)).min(1).values
        last = (c * ranks).max(1).values
        inp = R / last.clamp_min(1)
        tot["mADM"] += adm[valid].sum().item()
        tot["mAP"] += ap[valid].sum().item()
        tot["mINP"] += inp[valid].sum().item()
        for k in (1, 5, 10):
            tot[f"Rank-{k}"] += (first[valid] <= k).float().sum().item()
        n_q += int(valid.sum().item())
    return {k: v / max(n_q, 1) for k, v in tot.items()}


def par_metrics(probs, labels) -> dict:
    """Track 1 metrics (threshold 0.5), handy for the report."""
    pred = (probs > 0.5).float()
    gt = labels.float()
    tp, fp = (pred * gt).sum(0), (pred * (1 - gt)).sum(0)
    fn, tn = ((1 - pred) * gt).sum(0), ((1 - pred) * (1 - gt)).sum(0)
    ma = ((tp / (tp + fn).clamp_min(1) + tn / (tn + fp).clamp_min(1)) / 2).mean()
    inter, union = (pred * gt).sum(1), ((pred + gt) > 0).float().sum(1)
    prec = inter / pred.sum(1).clamp_min(1)
    rec = inter / gt.sum(1).clamp_min(1)
    f1 = (2 * prec * rec / (prec + rec).clamp_min(1e-12)).mean()
    return {"mA": ma.item(), "Inst_Acc": (inter / union.clamp_min(1)).mean().item(),
            "Inst_F1": f1.item(), "Challenge_Avg": (2 * ma * f1 / (ma + f1)).item()}
