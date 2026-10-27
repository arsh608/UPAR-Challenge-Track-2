"""Resize every image once to the model resolution and keep it as JPEG bytes in one file.

Colab has only 2 CPU cores, so decoding the original images every epoch would starve the
GPU. A 256x128 JPEG is ~8 KB, so the whole train+val set fits in ~1 GB of RAM.

    python prepare_cache.py --data-root ../data --out cache --size 256 128
"""
from __future__ import annotations

import argparse
import io
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image

_ROOT = ""
_SIZE = (256, 128)


def _encode(rel_path: str) -> bytes:
    img = Image.open(os.path.join(_ROOT, rel_path)).convert("RGB")
    img = img.resize((_SIZE[1], _SIZE[0]), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _init(root, size):
    global _ROOT, _SIZE
    _ROOT, _SIZE = root, size


def read_gt(csv_path: str):
    paths, labels = [], []
    with open(csv_path, "r", encoding="utf-8") as fh:
        next(fh)  # "# image,<40 attribute names>"
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 41:
                continue
            paths.append(parts[0])
            labels.append([int(float(v)) for v in parts[1:41]])
    return paths, np.asarray(labels, dtype=np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../data")
    ap.add_argument("--ann-dir", default=None, help="default: <data-root>/annotations")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--size", nargs=2, type=int, default=[256, 128], help="H W")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    ann = args.ann_dir or os.path.join(args.data_root, "annotations")
    os.makedirs(args.out, exist_ok=True)
    size = tuple(args.size)
    for split in args.splits:
        paths, labels = read_gt(os.path.join(ann, f"{split}.csv"))
        with Pool(args.workers, initializer=_init, initargs=(args.data_root, size)) as pool:
            blobs = []
            for i, b in enumerate(pool.imap(_encode, paths, chunksize=64)):
                blobs.append(b)
                if (i + 1) % 10000 == 0:
                    print(f"  {split}: {i + 1}/{len(paths)}", flush=True)
        lengths = np.fromiter((len(b) for b in blobs), dtype=np.int64, count=len(blobs))
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        np.save(os.path.join(args.out, f"{split}_jpg.npy"), np.frombuffer(b"".join(blobs), dtype=np.uint8))
        np.save(os.path.join(args.out, f"{split}_off.npy"), offsets)
        np.save(os.path.join(args.out, f"{split}_labels.npy"), labels)
        with open(os.path.join(args.out, f"{split}_paths.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(paths))
        print(f"{split}: {len(paths)} images, {offsets[-1] / 1e6:.0f} MB at {size[0]}x{size[1]}")


if __name__ == "__main__":
    main()
