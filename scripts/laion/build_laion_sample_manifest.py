#!/usr/bin/env python3
"""
Build **manifest.csv** + **MANIFEST.sha256** for a frozen ``laion_sample`` tree.

One row per retained ``(split, stem)`` pair (numeric ``*.jpg`` + matching ``*.txt``):

  split, stem, sha256_jpeg, caption_sha256, jpeg_bytes, caption_bytes, laion_meta

- ``sha256_jpeg`` / ``caption_sha256``: SHA-256 of **raw file bytes** on disk.
- ``laion_meta``: optional JSON/text provenance (see ``--stem-meta-csv``).

Rows are emitted in deterministic order: splits ``train``, ``train_calib``, ``val``, ``test``,
then ``stem`` lexicographically.

``MANIFEST.sha256`` records:

  - ``manifest_csv_sha256`` — SHA-256 of **exact bytes** of ``manifest.csv`` after write.
  - Optional ``sample_laion2b_en_1m_csv_sha256`` / path — if you pass ``--sample-csv``.
  - Optional ``repo_git_tag`` — pass ``--repo-tag`` or ``--git-describe``.

Usage::

  python scripts/laion/build_laion_sample_manifest.py \\
      --laion-root datasets/dataset_experiment/laion_sample \\
      --out-dir datasets/dataset_experiment/laion_sample \\
      --sample-csv data/laion2b_en_1m_sample.csv

Then archive ``manifest.csv``, ``MANIFEST.sha256``, and (if used) the reservoir CSV.

Optional enrichment: prepare ``stem_meta.csv`` with header::

  split,stem,laion_meta

where ``laion_meta`` is a single-line JSON string (escaped for CSV) copied from img2dataset /
your reservoir pipeline, keyed by whatever scheme you use to align stems (often manual or
custom join from retained parquet).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SPLITS = ("train", "train_calib", "val", "test")
SPLIT_ORDER = {s: i for i, s in enumerate(SPLITS)}


def _sha256_file(p: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def _load_stem_meta(path: Path | None) -> dict[tuple[str, str], str]:
    if path is None or not path.is_file():
        return {}
    out: dict[tuple[str, str], str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        need = {"split", "stem", "laion_meta"}
        if not r.fieldnames or not need.issubset(set(r.fieldnames)):
            raise ValueError(f"{path}: need columns {sorted(need)}, got {r.fieldnames}")
        for row in r:
            k = (row["split"].strip(), row["stem"].strip())
            out[k] = row.get("laion_meta", "") or ""
    return out


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def iter_rows(laion_root: Path, meta_map: dict[tuple[str, str], str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sp in SPLITS:
        d = laion_root / sp
        if not d.is_dir():
            continue
        for jpg in sorted(d.glob("*.jpg")):
            txt = jpg.with_suffix(".txt")
            if not txt.is_file():
                continue
            stem = jpg.stem
            jh, jb = _sha256_file(jpg)
            ch, cb = _sha256_file(txt)
            key = (sp, stem)
            lm = meta_map.get(key, "")
            rows.append(
                {
                    "split": sp,
                    "stem": stem,
                    "sha256_jpeg": jh,
                    "caption_sha256": ch,
                    "jpeg_bytes": jb,
                    "caption_bytes": cb,
                    "laion_meta": lm,
                }
            )
    rows.sort(key=lambda r: (SPLIT_ORDER[str(r["split"])], str(r["stem"])))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--laion-root",
        type=Path,
        default=Path("datasets/dataset_experiment/laion_sample"),
        help="Root containing train/, train_calib/, val/, test/",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write manifest.csv and MANIFEST.sha256 (default: --laion-root)",
    )
    ap.add_argument(
        "--stem-meta-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns split,stem,laion_meta to merge provenance",
    )
    ap.add_argument(
        "--sample-csv",
        type=Path,
        default=None,
        help="Optional reservoir CSV from sample_laion2b_en_1m.py — record SHA-256 in MANIFEST.sha256",
    )
    ap.add_argument(
        "--repo-tag",
        type=str,
        default="",
        help="Optional git tag / commit string to embed in MANIFEST.sha256 (e.g. paper-camera-ready)",
    )
    ap.add_argument(
        "--git-describe",
        action="store_true",
        help="Run `git describe --always --dirty` in cwd and record as repo_git_tag (overrides empty --repo-tag if succeeds)",
    )
    args = ap.parse_args()

    root = args.laion_root.resolve()
    out_dir = (args.out_dir or root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    checksum_path = out_dir / "MANIFEST.sha256"

    meta_map = _load_stem_meta(args.stem_meta_csv)
    rows = iter_rows(root, meta_map)

    buf = io.StringIO(newline="\n")
    fieldnames = [
        "split",
        "stem",
        "sha256_jpeg",
        "caption_sha256",
        "jpeg_bytes",
        "caption_bytes",
        "laion_meta",
    ]
    w = csv.DictWriter(buf, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    for row in rows:
        w.writerow(row)
    manifest_body = buf.getvalue()
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(manifest_body)

    m_hash = _sha256_bytes(manifest_body.encode("utf-8"))
    m_bytes = len(manifest_body.encode("utf-8"))

    lines = [
        "# LAION laion_sample manifest (deterministic row order)",
        f"manifest_csv_path={manifest_path}",
        f"manifest_csv_sha256={m_hash}",
        f"manifest_csv_bytes={m_bytes}",
        f"n_pairs={len(rows)}",
        f"generated_utc={datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    if args.sample_csv is not None:
        sp = args.sample_csv.resolve()
        if sp.is_file():
            sh, sz = _sha256_file(sp)
            lines.extend(
                [
                    "# Reservoir / img2dataset URL list (optional)",
                    f"sample_laion2b_en_1m_csv_path={sp}",
                    f"sample_laion2b_en_1m_csv_sha256={sh}",
                    f"sample_laion2b_en_1m_csv_bytes={sz}",
                    "",
                ]
            )
        else:
            print(f"WARNING: --sample-csv not found: {sp}", file=sys.stderr)

    tag = args.repo_tag.strip()
    if args.git_describe:
        try:
            tag = subprocess.check_output(
                ["git", "describe", "--always", "--dirty"],
                cwd=str(Path.cwd()),
                text=True,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    if tag:
        lines.extend(["# Code snapshot", f"repo_git_tag={tag}", ""])

    checksum_body = "\n".join(lines)
    if checksum_body:
        checksum_body += "\n"
    with open(checksum_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(checksum_body)

    print(f"Wrote {manifest_path} ({len(rows)} pairs)")
    print(f"Wrote {checksum_path}")
    print(f"manifest_csv_sha256={m_hash}")


if __name__ == "__main__":
    main()
