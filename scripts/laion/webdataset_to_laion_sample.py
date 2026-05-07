#!/usr/bin/env python3
"""
Convert img2dataset WebDataset shards (.tar) into the layout expected by this repository:

  <data_root>/laion_sample/<split>/*.jpg
  <data_root>/laion_sample/<split>/*.txt   (caption, UTF-8)

Same-stem .jpg / .txt pairs are required by `cache_embeddings_multi_gpu.py` (`process_laion_split`).

Caption is taken from img2dataset JSON sidecars: tries keys caption, TEXT, text, then first string value.

Usage:
  pip install webdataset  # recommended

  python scripts/laion/webdataset_to_laion_sample.py \\
      --shard-glob 'data/laion2b_en_1m_webdataset/*.tar' \\
      --data-root datasets/dataset_experiment \\
      --seed 42

Optional: only webdataset + stdlib — slow path uses tarfile if webdataset is missing.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import random
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SPLITS = ("train", "train_calib", "val", "test")


def _caption_from_json(raw: bytes) -> str:
    obj = json.loads(raw.decode("utf-8", errors="replace"))
    if isinstance(obj, str):
        return obj.strip()
    if not isinstance(obj, dict):
        return ""
    for k in ("caption", "TEXT", "text", "caption_txt"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for v in obj.values():
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _image_to_jpg_bytes(jpg: Any) -> Optional[bytes]:
    """Accept bytes or PIL.Image from webdataset decode."""
    if jpg is None:
        return None
    if isinstance(jpg, bytes):
        return jpg
    if isinstance(jpg, str):
        return jpg.encode("latin-1", errors="replace")
    try:
        from PIL import Image

        if isinstance(jpg, Image.Image):
            buf = io.BytesIO()
            rgb = jpg.convert("RGB")
            rgb.save(buf, format="JPEG", quality=95)
            return buf.getvalue()
    except Exception:
        pass
    return None


def _iter_pairs_webdataset(shard_paths: list[Path]) -> Iterator[Tuple[bytes, str]]:
    import webdataset as wds

    for sp in shard_paths:
        ds = wds.WebDataset(str(sp)).decode()
        for sample in ds:
            raw = (
                sample.get("jpg")
                or sample.get("jpeg")
                or sample.get("webp")
                or sample.get("png")
            )
            jpg_bytes = _image_to_jpg_bytes(raw)
            if jpg_bytes is None:
                continue
            cap = ""
            if "json" in sample:
                j = sample["json"]
                if isinstance(j, dict):
                    cap = _caption_from_json(json.dumps(j).encode("utf-8"))
                elif isinstance(j, bytes):
                    cap = _caption_from_json(j)
                else:
                    cap = _caption_from_json(str(j).encode("utf-8"))
            elif "txt" in sample:
                t = sample["txt"]
                cap = (
                    t.decode("utf-8", errors="replace").strip()
                    if isinstance(t, bytes)
                    else str(t).strip()
                )
            if not cap:
                continue
            yield jpg_bytes, cap


def _iter_pairs_tarfile(shard_paths: list[Path]) -> Iterator[Tuple[bytes, str]]:
    """Group .jpg and .json with the same basename inside each tar."""
    for sp in shard_paths:
        with tarfile.open(sp, "r:*") as tar:
            by_stem: Dict[str, Dict[str, bytes]] = {}
            for m in tar.getmembers():
                if not m.isfile():
                    continue
                name = m.name
                if "/" in name:
                    name = name.rsplit("/", 1)[-1]
                stem, dot, ext = name.partition(".")
                if not dot:
                    continue
                ext = ext.lower()
                if ext not in ("jpg", "jpeg", "json", "txt", "webp", "png"):
                    continue
                f = tar.extractfile(m)
                if f is None:
                    continue
                data = f.read()
                rec = by_stem.setdefault(stem, {})
                if ext in ("jpg", "jpeg", "webp", "png"):
                    rec["img"] = data
                    rec["img_ext"] = "jpg" if ext in ("jpg", "jpeg") else ext
                elif ext == "json":
                    rec["json"] = data
                elif ext == "txt":
                    rec["txt"] = data

            stems = sorted(by_stem.keys())
            for stem in stems:
                rec = by_stem[stem]
                img = rec.get("img")
                if img is None:
                    continue
                if "json" in rec:
                    cap = _caption_from_json(rec["json"])
                elif "txt" in rec:
                    cap = rec["txt"].decode("utf-8", errors="replace").strip()
                else:
                    continue
                if cap:
                    yield img, cap


def _assign_split(rng: random.Random, fracs: Tuple[float, float, float, float]) -> str:
    r = rng.random()
    acc = 0.0
    for s, f in zip(SPLITS, fracs):
        acc += f
        if r < acc:
            return s
    return SPLITS[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="WebDataset tars -> laion_sample jpg+txt (ψ-slack bundle)")
    ap.add_argument(
        "--shard-glob",
        type=str,
        required=True,
        help="Glob for .tar shards, e.g. data/out/*.tar",
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets/dataset_experiment"),
        help="Parent directory; writes laion_sample/ underneath",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--split-fracs",
        type=str,
        default="0.72,0.08,0.10,0.10",
        help="Comma fractions for train,train_calib,val,test (must sum to 1)",
    )
    ap.add_argument(
        "--use-tarfile",
        action="store_true",
        help="Force tarfile backend (no webdataset)",
    )
    args = ap.parse_args()

    fracs = tuple(float(x) for x in args.split_fracs.split(","))
    if len(fracs) != 4 or abs(sum(fracs) - 1.0) > 1e-6:
        raise SystemExit("--split-fracs must be four floats summing to 1.0")

    root = Path(args.data_root).resolve() / "laion_sample"
    for s in SPLITS:
        (root / s).mkdir(parents=True, exist_ok=True)

    import glob as glob_mod

    shard_paths = sorted(Path(p) for p in glob_mod.glob(args.shard_glob))
    shard_paths = [p for p in shard_paths if p.is_file()]
    if not shard_paths:
        raise FileNotFoundError(f"No shards matched: {args.shard_glob}")

    rng = random.Random(args.seed)
    use_wds = not args.use_tarfile
    if use_wds:
        try:
            import webdataset  # noqa: F401
        except ImportError:
            logger.warning("webdataset not installed; falling back to tarfile (pip install webdataset)")
            use_wds = False

    it = _iter_pairs_webdataset(shard_paths) if use_wds else _iter_pairs_tarfile(shard_paths)

    counts = {s: 0 for s in SPLITS}
    bad_img = 0
    for jpg_bytes, caption in it:
        try:
            from PIL import Image

            Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
        except Exception:
            bad_img += 1
            continue

        sp = _assign_split(rng, fracs)
        idx = counts[sp]
        counts[sp] += 1
        stem = f"{idx:08d}"
        jdir = root / sp
        jpath = jdir / f"{stem}.jpg"
        tpath = jdir / f"{stem}.txt"
        jpath.write_bytes(jpg_bytes)
        tpath.write_text(caption, encoding="utf-8")

        n = sum(counts.values())
        if n % 5000 == 0:
            logger.info("written %d samples (bad_img=%d) %s", n, bad_img, counts)

    logger.info("Done. totals=%s bad_img=%s root=%s", counts, bad_img, root)


if __name__ == "__main__":
    main()
