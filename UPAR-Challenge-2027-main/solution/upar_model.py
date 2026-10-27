"""CLIP ViT image encoder + attribute head, shared by training and the submission.

Plain torch only (no torchvision / open_clip), so the very same file runs inside the
Codabench container (torch 2.4.1, no network). Parameter names follow OpenAI CLIP's
visual tower, so the pretrained weights load directly.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

ARCHS = {
    "ViT-B-16": dict(width=768, layers=12, heads=12, patch=16, out_dim=512),
    "ViT-L-14": dict(width=1024, layers=24, heads=16, patch=14, out_dim=768),
}
DEFAULT_IMG = {"ViT-B-16": (256, 128), "ViT-L-14": (252, 126)}


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d, 4 * d)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(4 * d, d)),
        ]))

    def forward(self, x):
        y = self.ln_1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])
        self.grad_ckpt = False

    def forward(self, x):
        for blk in self.resblocks:
            x = checkpoint(blk, x, use_reentrant=False) if (self.grad_ckpt and self.training) else blk(x)
        return x


class VisionTransformer(nn.Module):
    """OpenAI CLIP visual tower on a non-square pedestrian grid (e.g. 256x128 -> 16x8 patches)."""

    def __init__(self, width, layers, heads, patch, out_dim, img_size=(256, 128)):
        super().__init__()
        self.patch = patch
        self.grid = (img_size[0] // patch, img_size[1] // patch)
        scale = width ** -0.5
        self.conv1 = nn.Conv2d(3, width, patch, patch, bias=False)
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid[0] * self.grid[1] + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, out_dim))  # kept so CLIP weights load strictly

    def forward(self, x):
        x = self.conv1(x).flatten(2).transpose(1, 2)  # B, N, C
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype)
        x = self.transformer(self.ln_pre(x))
        return self.ln_post(x)  # B, 1+N, C


class PARModel(nn.Module):
    """Takes RGB images in [0, 1] (B, 3, H, W) and returns 40 attribute logits."""

    def __init__(self, arch: str = "ViT-B-16", img_size=None, num_attrs: int = 40):
        super().__init__()
        self.arch = arch
        self.img_size = tuple(img_size or DEFAULT_IMG[arch])
        cfg = ARCHS[arch]
        self.backbone = VisionTransformer(**cfg, img_size=self.img_size)
        w = cfg["width"]
        self.head = nn.Sequential(nn.LayerNorm(2 * w), nn.Dropout(0.1), nn.Linear(2 * w, num_attrs))
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        x = (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)
        tok = self.backbone(x)
        feat = torch.cat([tok[:, 0], tok[:, 1:].mean(1)], dim=1)  # global token + pooled patches
        return self.head(feat)


def load_clip_visual(model: PARModel, clip_state: dict) -> None:
    """Load OpenAI CLIP visual weights, resampling the square position grid to ours."""
    sd = {k.removeprefix("visual."): v for k, v in clip_state.items()}
    pe = sd["positional_embedding"].float()
    cls, grid = pe[:1], pe[1:]
    n = int(math.sqrt(grid.shape[0]))
    if (n, n) != model.backbone.grid:
        grid = grid.reshape(1, n, n, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=model.backbone.grid, mode="bicubic", align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(-1, pe.shape[1])
    sd["positional_embedding"] = torch.cat([cls, grid])
    sd = {k: v.float() for k, v in sd.items()}
    missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"CLIP weights mismatch: missing={missing} unexpected={unexpected}")


def param_groups(model: PARModel, lr: float, head_lr: float, wd: float, layer_decay: float):
    """AdamW groups with layer-wise lr decay: early CLIP layers move least."""
    n_layers = len(model.backbone.transformer.resblocks)
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("head."):
            depth, base = None, head_lr
        elif ".resblocks." in name:
            depth = int(name.split(".resblocks.")[1].split(".")[0]) + 1
        elif any(s in name for s in ("ln_post", "backbone.proj")):
            depth = n_layers + 1
        else:  # conv1, embeddings, ln_pre
            depth = 0
        if depth is not None:
            base = lr * layer_decay ** (n_layers + 1 - depth)
        no_wd = p.ndim < 2 or "embedding" in name
        key = (base, no_wd)
        g = groups.setdefault(key, {"params": [], "lr": base, "weight_decay": 0.0 if no_wd else wd})
        g["params"].append(p)
    return list(groups.values())


def save_checkpoint(model: PARModel, path: str, extra: dict | None = None) -> None:
    sd = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save({"arch": model.arch, "img_size": list(model.img_size), "state_dict": sd, **(extra or {})}, path)


def load_checkpoint(path: str, map_location="cpu") -> PARModel:
    ck = torch.load(path, map_location=map_location, weights_only=True)
    model = PARModel(ck["arch"], tuple(ck["img_size"]))
    model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    return model
