#!/usr/bin/env python3
"""
Check cached ``laion_sample_*_{image,text}.npz`` files against ``paper_draft/info.md``
defaults for the primary LAION backbone (OpenCLIP **CLIP-ViT-L-14** LAION-2B).

Verifies per-file ``metadata`` (model id, dim, dtype, library), ``ids`` naming
(``laion_sample_{split}_{i}`` / ``…_cap0``), dtype/shape, and optional L2 norms (~1).

Run from repository root (``psi_slack/``)::

  python scripts/verify_laion_embeddings_metadata.py --embed-dir embeddings_laion

  # Optional: only check splits you cached
  python scripts/verify_laion_embeddings_metadata.py --embed-dir embeddings_laion \\
      --splits train,train_calib,val,test
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# Defaults documented in paper_draft/info.md §2 (LAION CLIP)
DEFAULT_EXPECTED = {
    "model_name": "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "embedding_dim": 768,
    "library": "open_clip",
    "dtype": "torch.float32",
}

IMAGE_ID_RE = re.compile(r"^laion_sample_(train|train_calib|val|test)_(\d+)$")
TEXT_ID_RE = re.compile(r"^laion_sample_(train|train_calib|val|test)_(\d+)_cap0$")


def _meta_dict(raw) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, np.ndarray) and raw.ndim == 0:
        v = raw.item()
        return dict(v) if isinstance(v, dict) else {}
    if isinstance(raw, dict):
        return raw
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--embed-dir",
        type=Path,
        default=Path("embeddings_laion"),
        help="Directory containing laion_sample_*_{image,text}.npz",
    )
    ap.add_argument(
        "--splits",
        type=str,
        default="train,train_calib,val,test",
        help="Comma-separated splits to require",
    )
    ap.add_argument(
        "--check-norms",
        action="store_true",
        help="Sample up to N rows and check L2 norms ≈ 1 (normalized embeddings)",
    )
    ap.add_argument("--norm-sample", type=int, default=2048, help="Max rows to norm-check per file")
    ap.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=f"Override expected metadata model_name (default: {DEFAULT_EXPECTED['model_name']})",
    )
    args = ap.parse_args()

    embed_dir = args.embed_dir.resolve()
    if not embed_dir.is_dir():
        print(f"ERROR: embed dir not found: {embed_dir}", file=sys.stderr)
        return 1

    expected = dict(DEFAULT_EXPECTED)
    if args.model_name:
        expected["model_name"] = args.model_name.strip()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    errors: list[str] = []

    for split in splits:
        for kind, id_re in [("image", IMAGE_ID_RE), ("text", TEXT_ID_RE)]:
            path = embed_dir / f"laion_sample_{split}_{kind}.npz"
            if not path.is_file():
                errors.append(f"missing file: {path}")
                continue
            z = np.load(path, allow_pickle=True)
            try:
                emb = np.asarray(z["embeddings"])
                ids = np.asarray(z["ids"], dtype=object)
                meta = _meta_dict(z.get("metadata"))

                if emb.dtype != np.float32:
                    errors.append(f"{path}: embeddings dtype {emb.dtype}, expected float32")
                if emb.ndim != 2:
                    errors.append(f"{path}: embeddings ndim {emb.ndim}, expected 2")
                dm = meta.get("embedding_dim")
                if dm is not None and int(dm) != expected["embedding_dim"]:
                    errors.append(
                        f"{path}: metadata embedding_dim={dm}, expected {expected['embedding_dim']}"
                    )
                if emb.shape[1] != expected["embedding_dim"]:
                    errors.append(
                        f"{path}: embedding dim {emb.shape[1]}, expected {expected['embedding_dim']}"
                    )

                mn = meta.get("model_name")
                if mn != expected["model_name"]:
                    errors.append(
                        f"{path}: metadata model_name={mn!r}, expected {expected['model_name']!r}"
                    )
                lib = meta.get("library")
                if lib != expected["library"]:
                    errors.append(
                        f"{path}: metadata library={lib!r}, expected {expected['library']!r}"
                    )
                dt = meta.get("dtype")
                if dt != expected["dtype"]:
                    errors.append(
                        f"{path}: metadata dtype={dt!r}, expected {expected['dtype']!r}"
                    )

                if len(ids) != len(emb):
                    errors.append(f"{path}: len(ids)={len(ids)} != rows={len(emb)}")
                for j, rid in enumerate(ids[: min(5000, len(ids))]):
                    s = str(rid)
                    if not id_re.match(s):
                        errors.append(f"{path}: bad id[{j}]={s!r}")
                        break

                if args.check_norms and len(emb) > 0:
                    n = min(args.norm_sample, len(emb))
                    rows = emb[:n]
                    norms = np.linalg.norm(rows, axis=1)
                    dev = np.abs(norms - 1.0).max()
                    if dev > 0.02:
                        errors.append(
                            f"{path}: L2 norms deviate from 1 (max |‖x‖-1|={dev:.4f}, checked {n} rows)"
                        )
            finally:
                z.close()

    if errors:
        print("FAILED checks:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"OK — all checked NPZs under {embed_dir} match info.md LAION CLIP defaults.")
    print(f"    model_name={expected['model_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
