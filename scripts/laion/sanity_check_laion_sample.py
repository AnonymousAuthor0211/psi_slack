#!/usr/bin/env python3
"""
Quick sanity checks: original sampled metadata (CSV) vs retained laion_sample on disk.

1) Distribution comparison (unpaired): caption length; similarity / width / height / language / URL domain
   when available in --original-csv. Retained side uses optional --retained-metadata (parquet/csv from
   img2dataset successes) OR filesystem (.txt + optional PIL image size).

2) Splits: disjoint paths (by construction), exact duplicate images across splits (SHA256 of file bytes).

3) Test split size: counts and a short note on retrieval stability.

Usage:
  pip install pandas pyarrow pillow tqdm
  # optional: scipy (KS test), numpy

  python scripts/laion/sanity_check_laion_sample.py \\
    --original-csv data/laion2b_en_1m_sample.csv \\
    --laion-root datasets/dataset_experiment/laion_sample \\
    --retained-metadata data/img2dataset_success.parquet

If --retained-metadata is omitted, retained caption lengths and image sizes come from laion_sample only;
similarity / domain comparisons on the retained side are skipped (with a clear note).

By default the same log lines are written to:

  <laion-root>/SANITY_CHECK_REPORT.txt

Use --no-save-report for terminal only, or --output-report PATH to choose the file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


SPLITS = ("train", "train_calib", "val", "test")


def _describe(name: str, x: np.ndarray) -> None:
    x = x[np.isfinite(x)]
    if x.size == 0:
        logger.info("%s: (empty)", name)
        return
    qs = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    logger.info(
        "%s: n=%s mean=%.4g std=%.4g "
        "p5=%.4g p25=%.4g p50=%.4g p75=%.4g p95=%.4g p99=%.4g",
        name,
        f"{x.size:,}",
        float(np.mean(x)),
        float(np.std(x)),
        *map(float, qs),
    )


def _ks_report(a: np.ndarray, b: np.ndarray, label: str) -> None:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        logger.info("KS %s: skip (empty array)", label)
        return
    if scipy_stats is None:
        logger.info("KS %s: install scipy for ks_2samp", label)
        return
    stat, p = scipy_stats.ks_2samp(a, b)
    logger.info("KS %s: statistic=%.6f p-value=%.4g (two-sample, unpaired)", label, stat, p)


def _domain(url: Any) -> str:
    if url is None or (isinstance(url, float) and np.isnan(url)):
        return ""
    s = str(url).strip()
    if not s:
        return ""
    if "://" not in s:
        s = "http://" + s
    try:
        return urlparse(s).netloc.lower() or ""
    except Exception:
        return ""


def _top_domains(urls: Sequence[Any], k: int = 15) -> List[Tuple[str, int]]:
    c = Counter(_domain(u) for u in urls if _domain(u))
    return c.most_common(k)


def _parse_laion_meta_cell(cell: Any) -> Dict[str, Any]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return {}
    if isinstance(cell, dict):
        return cell
    s = str(cell).strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}


def _meta_float(m: Dict[str, Any], *keys: str) -> float:
    for k in keys:
        if k not in m or m[k] is None:
            continue
        try:
            return float(m[k])
        except (TypeError, ValueError):
            return np.nan
    return np.nan


def load_original_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # url / caption column names
    url_c = "url" if "url" in df.columns else ("URL" if "URL" in df.columns else None)
    cap_c = (
        "caption"
        if "caption" in df.columns
        else ("TEXT" if "TEXT" in df.columns else ("text" if "text" in df.columns else None))
    )
    if url_c is None or cap_c is None:
        raise ValueError(f"CSV needs url + caption columns. Have: {list(df.columns)}")
    df = df.rename(columns={url_c: "_url", cap_c: "_caption"})
    df["_cap_len"] = df["_caption"].astype(str).str.len()

    # Unpack laion_meta JSON if present
    if "laion_meta" in df.columns:
        metas = [_parse_laion_meta_cell(x) for x in df["laion_meta"]]
        df["_meta_sim"] = [_meta_float(m, "similarity", "Similarity") for m in metas]
        df["_meta_w"] = [_meta_float(m, "WIDTH", "width") for m in metas]
        df["_meta_h"] = [_meta_float(m, "HEIGHT", "height") for m in metas]
        df["_meta_lang"] = [
            str(m.get("LANGUAGE") or m.get("language") or "") for m in metas
        ]
    else:
        for col, out in (
            ("similarity", "_meta_sim"),
            ("width", "_meta_w"),
            ("WIDTH", "_meta_w"),
            ("height", "_meta_h"),
            ("HEIGHT", "_meta_h"),
            ("LANGUAGE", "_meta_lang"),
            ("language", "_meta_lang"),
        ):
            if col in df.columns and out not in df.columns:
                if out == "_meta_lang":
                    df[out] = df[col].astype(str).replace("nan", "")
                else:
                    df[out] = pd.to_numeric(df[col], errors="coerce")

    df["_domain"] = [_domain(u) for u in df["_url"]]
    return df


def load_retained_metadata(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    url_c = next((c for c in ("url", "URL") if c in df.columns), None)
    cap_c = next((c for c in ("caption", "TEXT", "text") if c in df.columns), None)
    if cap_c is None:
        raise ValueError(f"Retained metadata needs a caption column: {list(df.columns)}")
    out = pd.DataFrame()
    if url_c:
        out["_url"] = df[url_c]
        out["_domain"] = [_domain(u) for u in out["_url"]]
    out["_cap_len"] = df[cap_c].astype(str).str.len()
    if "similarity" in df.columns:
        out["_meta_sim"] = pd.to_numeric(df["similarity"], errors="coerce")
    if "width" in df.columns:
        out["_meta_w"] = pd.to_numeric(df["width"], errors="coerce")
    elif "WIDTH" in df.columns:
        out["_meta_w"] = pd.to_numeric(df["WIDTH"], errors="coerce")
    if "height" in df.columns:
        out["_meta_h"] = pd.to_numeric(df["height"], errors="coerce")
    elif "HEIGHT" in df.columns:
        out["_meta_h"] = pd.to_numeric(df["HEIGHT"], errors="coerce")
    if "LANGUAGE" in df.columns:
        out["_meta_lang"] = df["LANGUAGE"].astype(str)
    elif "language" in df.columns:
        out["_meta_lang"] = df["language"].astype(str)
    return out


def iter_laion_pairs(laion_root: Path) -> Iterable[Tuple[str, Path, Path]]:
    root = laion_root
    if not root.is_dir():
        raise FileNotFoundError(laion_root)
    for sp in SPLITS:
        d = root / sp
        if not d.is_dir():
            continue
        for jpg in sorted(d.glob("*.jpg")):
            txt = jpg.with_suffix(".txt")
            if txt.is_file():
                yield sp, jpg, txt


def scan_filesystem_retained(
    laion_root: Path,
    max_read_images: Optional[int],
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    all_pairs = list(iter_laion_pairs(laion_root))
    total_pairs = len(all_pairs)
    pairs = all_pairs
    rng = np.random.default_rng(seed)
    if max_read_images is not None and len(pairs) > max_read_images:
        idx = rng.choice(len(pairs), size=max_read_images, replace=False)
        pairs = [pairs[i] for i in sorted(idx)]
        logger.info(
            "Subsampled %s / %s jpg+txt pairs for PIL size read",
            len(pairs),
            total_pairs,
        )

    try:
        from PIL import Image
    except ImportError:
        Image = None
        logger.warning("Pillow not installed; image width/height skipped for filesystem scan")

    for sp, jpg, txt in tqdm(pairs, desc="laion_sample files"):
        cap = txt.read_text(encoding="utf-8", errors="replace")
        w = h = np.nan
        if Image is not None:
            try:
                with Image.open(jpg) as im:
                    w, h = im.size
            except Exception:
                pass
        rows.append(
            {
                "split": sp,
                "jpg": str(jpg),
                "_cap_len": len(cap),
                "_img_w": w,
                "_img_h": h,
            }
        )
    return pd.DataFrame(rows)


def check_path_disjoint(laion_root: Path) -> None:
    all_paths: Dict[str, str] = {}
    for sp, jpg, txt in iter_laion_pairs(laion_root):
        for p in (jpg, txt):
            key = str(p.resolve())
            if key in all_paths:
                raise RuntimeError(f"Duplicate path {key}")
            all_paths[key] = sp
    # Same logical file cannot appear in two splits
    by_stem: Dict[Tuple[str, str], str] = {}
    for sp, jpg, txt in iter_laion_pairs(laion_root):
        stem = jpg.stem
        k = (sp, stem)
        if k in by_stem:
            raise RuntimeError(f"Duplicate stem in split {sp}: {stem}")
        by_stem[k] = sp
    logger.info("Path disjointness: OK (%s unique image paths across splits)", len(by_stem))


def check_duplicate_image_bytes(laion_root: Path) -> Tuple[int, List[Tuple[str, List[str]]]]:
    """Return (n_jpg, list of (hexdigest, [paths...]) for hashes with >1 file)."""
    hmap: Dict[str, List[str]] = {}
    n = 0
    for sp, jpg, txt in tqdm(list(iter_laion_pairs(laion_root)), desc="hash images"):
        n += 1
        data = jpg.read_bytes()
        h = hashlib.sha256(data).hexdigest()
        hmap.setdefault(h, []).append(f"{sp}/{jpg.name}")
    dups = [(h, paths) for h, paths in hmap.items() if len(paths) > 1]
    return n, dups


def sample_near_duplicate_phash(
    laion_root: Path,
    n_sample: int,
    hamming_max: int,
    seed: int,
) -> None:
    """Cheap heuristic: random sample, pairwise phash Hamming <= hamming_max (not exhaustive)."""
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        logger.info("Near-dup check skipped (pip install imagehash pillow)")
        return
    pairs = list(iter_laion_pairs(laion_root))
    rng = np.random.default_rng(seed)
    if len(pairs) > n_sample:
        idx = rng.choice(len(pairs), size=n_sample, replace=False)
        pairs = [pairs[i] for i in idx]
    hashes = []
    for sp, jpg, txt in tqdm(pairs, desc="phash sample"):
        try:
            with Image.open(jpg) as im:
                h = imagehash.phash(im.convert("RGB"))
            hashes.append((h, f"{sp}/{jpg.name}"))
        except Exception:
            continue
    n_hit = 0
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            d = hashes[i][0] - hashes[j][0]
            if d <= hamming_max:
                n_hit += 1
                if n_hit <= 15:
                    logger.warning(
                        "Near-dup (phash Hamming<=%s): %s <-> %s (d=%s)",
                        hamming_max,
                        hashes[i][1],
                        hashes[j][1],
                        d,
                    )
    logger.info(
        "Near-dup sample: %s images, %s pairs with phash Hamming<=%s (exhaustive within sample only)",
        len(hashes),
        n_hit,
        hamming_max,
    )


def report_test_size(laion_root: Path) -> None:
    counts = {sp: len(list((laion_root / sp).glob("*.jpg"))) for sp in SPLITS if (laion_root / sp).is_dir()}
    nt = counts.get("test", 0)
    logger.info("Per-split image counts: %s", counts)
    logger.info(
        "Test split: n=%s. For retrieval R@K, variance scales ~1/sqrt(n_queries); "
        "tens of thousands of queries is usually ample for stable headline metrics.",
        f"{nt:,}",
    )


def compare_top_domains(orig_domains: List[str], retr_domains: List[str], k: int = 12) -> None:
    orig_domains = [d for d in orig_domains if d]
    retr_domains = [d for d in retr_domains if d]
    co = Counter(orig_domains)
    cr = Counter(retr_domains)
    keys = set(co) | set(cr)
    # share of mass in top-k orig
    tot_o = sum(co.values())
    tot_r = sum(cr.values())
    logger.info("Domain coverage: original unique=%s retained unique=%s", f"{len(co):,}", f"{len(cr):,}")
    logger.info("Top domains (original):")
    for dom, c in co.most_common(k):
        logger.info("  %s  %s (%.2f%%)", dom, f"{c:,}", 100.0 * c / tot_o if tot_o else 0)
    logger.info("Top domains (retained):")
    for dom, c in cr.most_common(k):
        logger.info("  %s  %s (%.2f%%)", dom, f"{c:,}", 100.0 * c / tot_r if tot_r else 0)
    # L1 diff on shared top domains
    top = [d for d, _ in co.most_common(50)]
    diff = sum(abs(co.get(d, 0) / tot_o - cr.get(d, 0) / tot_r) for d in top) / 2.0
    logger.info("Approx L1 distance over top-50 original domains (normalized freqs): %.4f", diff)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sanity check original LAION CSV vs retained laion_sample")
    ap.add_argument("--original-csv", type=Path, required=True)
    ap.add_argument("--laion-root", type=Path, default=Path("datasets/dataset_experiment/laion_sample"))
    ap.add_argument(
        "--retained-metadata",
        type=Path,
        default=None,
        help="Parquet/CSV of successful downloads (same schema as img2dataset output) for full retained stats",
    )
    ap.add_argument(
        "--max-read-images",
        type=int,
        default=None,
        help="Subsample this many random pairs when reading PIL sizes from disk (default: all)",
    )
    ap.add_argument(
        "--near-dup-phash-sample",
        type=int,
        default=0,
        help="If >0, sample this many images and report pairs with perceptual hash Hamming <= --near-dup-hamming",
    )
    ap.add_argument("--near-dup-hamming", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Write the same log output to this file (default: <laion-root>/SANITY_CHECK_REPORT.txt)",
    )
    ap.add_argument(
        "--no-save-report",
        action="store_true",
        help="Do not write a report file; only print to the terminal",
    )
    args = ap.parse_args()

    report_path: Optional[Path] = None
    if not args.no_save_report:
        report_path = (
            args.output_report
            if args.output_report is not None
            else args.laion_root.resolve() / "SANITY_CHECK_REPORT.txt"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(report_path, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        fh.setLevel(logging.INFO)
        logging.getLogger().addHandler(fh)
        print(
            f"Sanity check report will be saved to: {report_path}",
            file=sys.stderr,
        )

    logger.info(
        "Started %s | original-csv=%s | laion-root=%s",
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        args.original_csv,
        args.laion_root,
    )

    logger.info("Loading original CSV: %s", args.original_csv)
    orig = load_original_csv(args.original_csv)
    logger.info("Original rows: %s", f"{len(orig):,}")

    _describe("original caption_len", orig["_cap_len"].to_numpy(dtype=float))
    for col, label in (
        ("_meta_sim", "original similarity"),
        ("_meta_w", "original width"),
        ("_meta_h", "original height"),
    ):
        if col in orig.columns and orig[col].notna().any():
            _describe(label, orig[col].to_numpy(dtype=float))

    if "_meta_lang" in orig.columns and orig["_meta_lang"].astype(str).str.len().sum() > 0:
        lc = orig["_meta_lang"].astype(str).value_counts().head(10)
        logger.info("Original LANGUAGE top-10:\n%s", lc.to_string())

    td = _top_domains(orig["_url"].tolist(), 12)
    logger.info("Original URL top domains: %s", td)

    logger.info("--- Filesystem retained (laion_sample) ---")
    fs_df = scan_filesystem_retained(args.laion_root, args.max_read_images, args.seed)
    logger.info("Retained pairs on disk: %s", f"{len(fs_df):,}")
    _describe("retained (fs) caption_len", fs_df["_cap_len"].to_numpy(dtype=float))
    if fs_df["_img_w"].notna().any():
        _describe("retained (fs) image width (PIL)", fs_df["_img_w"].to_numpy(dtype=float))
        _describe("retained (fs) image height (PIL)", fs_df["_img_h"].to_numpy(dtype=float))

    _ks_report(orig["_cap_len"].to_numpy(dtype=float), fs_df["_cap_len"].to_numpy(dtype=float), "caption_len orig vs fs")

    retr_meta = None
    if args.retained_metadata and args.retained_metadata.is_file():
        logger.info("Loading retained metadata: %s", args.retained_metadata)
        retr_meta = load_retained_metadata(args.retained_metadata)
        logger.info("Retained metadata rows: %s", f"{len(retr_meta):,}")
        _describe("retained (meta) caption_len", retr_meta["_cap_len"].to_numpy(dtype=float))
        _ks_report(
            orig["_cap_len"].to_numpy(dtype=float),
            retr_meta["_cap_len"].to_numpy(dtype=float),
            "caption_len orig vs retained-meta",
        )
        for col, label in (
            ("_meta_sim", "similarity"),
            ("_meta_w", "width"),
            ("_meta_h", "height"),
        ):
            if col in retr_meta.columns and retr_meta[col].notna().any() and col in orig.columns:
                _describe(f"retained (meta) {label}", retr_meta[col].to_numpy(dtype=float))
                _ks_report(orig[col].to_numpy(dtype=float), retr_meta[col].to_numpy(dtype=float), label)
        if "_domain" in retr_meta.columns and retr_meta["_domain"].astype(str).str.len().sum() > 0:
            compare_top_domains(orig["_domain"].tolist(), retr_meta["_domain"].tolist())
        elif "_url" in retr_meta.columns:
            compare_top_domains(orig["_domain"].tolist(), [_domain(u) for u in retr_meta["_url"]])
        if "_meta_lang" in retr_meta.columns and "_meta_lang" in orig.columns:
            if retr_meta["_meta_lang"].astype(str).str.strip().str.len().gt(0).any():
                logger.info(
                    "LANGUAGE retained top-10:\n%s",
                    retr_meta["_meta_lang"].astype(str).value_counts().head(10).to_string(),
                )
    else:
        logger.info(
            "No --retained-metadata: skip retained-side similarity / domain / metadata width-height "
            "(export img2dataset success parquet to enable)."
        )

    logger.info("--- Split / duplicate checks ---")
    check_path_disjoint(args.laion_root)
    n_jpg, dups = check_duplicate_image_bytes(args.laion_root)
    if dups:
        ndup_files = sum(len(p) for _, p in dups)
        logger.warning(
            "EXACT duplicate image bytes: %s hash groups (%s files), total jpg hashed=%s",
            len(dups),
            ndup_files,
            f"{n_jpg:,}",
        )
        for h, paths in dups[:20]:
            logger.warning("  hash=%s... paths=%s", h[:16], paths)
        if len(dups) > 20:
            logger.warning("  ... and %s more duplicate groups", len(dups) - 20)
    else:
        logger.info("EXACT duplicate image bytes across splits: none (%s files hashed)", f"{n_jpg:,}")

    if args.near_dup_phash_sample > 0:
        sample_near_duplicate_phash(
            args.laion_root,
            args.near_dup_phash_sample,
            args.near_dup_hamming,
            args.seed,
        )

    report_test_size(args.laion_root)

    logger.info("Done.")
    if report_path is not None:
        logger.info("Full report file: %s", report_path.resolve())
        print(f"\nFull report saved to: {report_path.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
