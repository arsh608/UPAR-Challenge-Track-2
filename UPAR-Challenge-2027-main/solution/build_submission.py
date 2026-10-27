"""Pack the Codabench Track 2 zip (run.py at the archive root).

    python build_submission.py --ckpt runs/final/model_ema.pt --scoring runs/dev/scoring.json --out upar_t2.zip
    # ensemble: pass several --ckpt
"""
from __future__ import annotations

import argparse
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
METADATA = os.path.join(HERE, "..", "examples", "task2", "sample_code_submission", "metadata.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--scoring", required=True)
    ap.add_argument("--out", default="upar_t2_submission.zip")
    args = ap.parse_args()
    files = {
        "run.py": os.path.join(HERE, "submission_src", "run.py"),
        "upar_model.py": os.path.join(HERE, "upar_model.py"),
        "retrieval.py": os.path.join(HERE, "retrieval.py"),
        "metadata.yaml": METADATA,
        "assets/scoring.json": args.scoring,
    }
    for i, ck in enumerate(args.ckpt):
        files[f"assets/model{i}.pt"] = ck
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_STORED) as zf:
        for arc, src in files.items():
            zf.write(src, arc)
            print(f"  {arc:24s} <- {src}")
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
