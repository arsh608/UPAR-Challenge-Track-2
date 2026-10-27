"""Cached-JPEG dataset and GPU-side augmentation for training."""
from __future__ import annotations

import io
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class CacheDataset(Dataset):
    """Reads the files written by prepare_cache.py. Returns (uint8 3xHxW, float labels)."""

    def __init__(self, cache_dir: str, splits: list[str]):
        self.blobs, self.offs, labels, self.paths = [], [], [], []
        for s in splits:
            self.blobs.append(np.load(os.path.join(cache_dir, f"{s}_jpg.npy")))
            self.offs.append(np.load(os.path.join(cache_dir, f"{s}_off.npy")))
            labels.append(np.load(os.path.join(cache_dir, f"{s}_labels.npy")))
            with open(os.path.join(cache_dir, f"{s}_paths.txt"), encoding="utf-8") as fh:
                self.paths += fh.read().split("\n")
        self.index = [(i, j) for i, lab in enumerate(labels) for j in range(len(lab))]
        self.labels = torch.from_numpy(np.concatenate(labels).astype(np.float32))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, j = self.index[k]
        raw = self.blobs[i][self.offs[i][j]:self.offs[i][j + 1]].tobytes()
        img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
        return torch.from_numpy(img).permute(2, 0, 1).contiguous(), self.labels[k]


def gpu_augment(x: torch.Tensor, low_res_p: float = 0.3, erase_p: float = 0.3) -> torch.Tensor:
    """x: uint8 (B,3,H,W) on GPU -> float [0,1]. No hue changes: colour is a label here.

    The low-resolution simulation (down- then up-sample) mimics distant surveillance
    cameras, the main domain gap to the hidden test set.
    """
    x = x.float() / 255.0
    B, _, H, W = x.shape
    dev = x.device
    # horizontal flip (no left/right attributes in the vocabulary)
    flip = torch.rand(B, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    # random scale + translation
    s = torch.empty(B, device=dev).uniform_(0.85, 1.1)
    asp = torch.empty(B, device=dev).uniform_(0.9, 1.1)
    tx, ty = (torch.rand(B, device=dev) - 0.5) * 0.16, (torch.rand(B, device=dev) - 0.5) * 0.16
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 1, 1] = 1 / (s * asp), 1 / s
    theta[:, 0, 2], theta[:, 1, 2] = tx, ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    # low-resolution simulation
    lr = torch.rand(B, device=dev) < low_res_p
    if lr.any():
        f = float(np.random.uniform(0.25, 0.6))
        small = F.interpolate(x[lr], scale_factor=f, mode="bilinear", align_corners=False, antialias=True)
        x[lr] = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
    # brightness / contrast / mild saturation
    b = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.75, 1.25)
    c = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.75, 1.25)
    sat = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.8, 1.2)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x * b - m) * c + m
    gray = x.mean(1, keepdim=True)
    x = (x - gray) * sat + gray
    # small random erasing (kept small so accessories are rarely wiped out)
    er = torch.rand(B, device=dev) < erase_p
    for k in torch.nonzero(er).flatten().tolist():
        area = np.random.uniform(0.02, 0.12) * H * W
        r = np.random.uniform(0.3, 3.3)
        eh, ew = int(min(H, (area * r) ** 0.5)), int(min(W, (area / r) ** 0.5))
        y0, x0 = np.random.randint(0, H - eh + 1), np.random.randint(0, W - ew + 1)
        x[k, :, y0:y0 + eh, x0:x0 + ew] = torch.rand(3, 1, 1, device=dev)
    return x.clamp(0, 1)


def read_queries(csv_path: str) -> np.ndarray:
    return np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
