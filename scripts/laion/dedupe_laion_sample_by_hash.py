#!/usr/bin/env python3
"""
Deduplicate laion_sample by exact image bytes (SHA256).

For each hash group, assign a single canonical split using priority:
  train > train_calib > val > test

Exactly one (jpg, txt) pair per hash remains; all other copies are removed.
If no copy lies in the winning split yet, one pair is moved there (new numeric stem).

Usage:
  python scripts/laion/dedupe_laion_sample_by_hash.py \\
    --laion-root datasets/dataset_experiment/laion_sample --dry-run

  python scripts/laion/dedupe_laion_sample_by_hash.py \\
    --laion-root datasets/dataset_experiment/laion_sample
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x

SPLITS = ("train", "train_calib", "val", "test")
SPLIT_PRIORITY = {s: i for i, s in enumerate(SPLITS)}

# (split, jpg_path, txt_path)
Loc = Tuple[str, Path, Path]


def iter_pairs(laion_root: Path) -> List[Loc]:
    out: List[Loc] = []
    for sp in SPLITS:
        d = laion_root / sp
        if not d.is_dir():
            continue
        for jpg in sorted(d.glob("*.jpg")):
            txt = jpg.with_suffix(".txt")
            if txt.is_file():
                out.append((sp, jpg.resolve(), txt.resolve()))
    return out


def hash_jpg(jpg: Path) -> str:
    h = hashlib.sha256()
    with open(jpg, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def max_numeric_stem(split_dir: Path) -> int:
    m = 0
    if not split_dir.is_dir():
        return 0
    for p in split_dir.glob("*.jpg"):
        if p.stem.isdigit():
            m = max(m, int(p.stem))
    return m


def choose_target_split(splits_in_group: List[str]) -> str:
    return min(splits_in_group, key=lambda s: SPLIT_PRIORITY[s])


def main() -> None:
    ap = argparse.ArgumentParser(description="Dedupe laion_sample by SHA256; one split per hash")
    ap.add_argument("--laion-root", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="Print actions only")
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Append summary to this file (default: laion-root/DEDUPE_HASH_REPORT.txt)",
    )
    args = ap.parse_args()

    root = args.laion_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    report_path = args.report if args.report is not None else root / "DEDUPE_HASH_REPORT.txt"

    logger.info("Scanning %s", root)
    pairs = iter_pairs(root)
    if not pairs:
        raise SystemExit("No jpg+txt pairs found.")

    hash_to_locs: Dict[str, List[Loc]] = defaultdict(list)
    for sp, jpg, txt in tqdm(pairs, desc="hash images"):
        h = hash_jpg(jpg)
        hash_to_locs[h].append((sp, jpg, txt))

    multi = {h: locs for h, locs in hash_to_locs.items() if len(locs) > 1}
    logger.info(
        "Unique hashes: %s | duplicate groups: %s | pairs in groups: %s",
        f"{len(hash_to_locs):,}",
        f"{len(multi):,}",
        f"{sum(len(v) for v in multi.values()):,}",
    )

    # Next stem per split (lazily updated)
    next_stem: Dict[str, int] = {}
    for sp in SPLITS:
        next_stem[sp] = max_numeric_stem(root / sp) + 1

    def allocate_stem(sp: str) -> str:
        n = next_stem[sp]
        next_stem[sp] = n + 1
        return f"{n:08d}"

    n_delete_pairs = 0  # number of (jpg,txt) pairs removed
    n_move = 0
    n_groups_fixed = 0
    cross_split_examples: List[str] = []

    lines: List[str] = []

    for h in sorted(multi.keys()):
        locs = multi[h]
        splits_in_group = [loc[0] for loc in locs]
        unique_splits = list(dict.fromkeys(splits_in_group))
        target = choose_target_split(unique_splits)
        in_target = [loc for loc in locs if loc[0] == target]

        if len(unique_splits) > 1:
            cross_split_examples.append(
                f"hash={h[:16]}... target={target} splits={unique_splits} n={len(locs)}"
            )

        if in_target:
            keeper = sorted(in_target, key=lambda x: str(x[1]))[0]
        else:
            keeper = sorted(locs, key=lambda x: (SPLIT_PRIORITY[x[0]], str(x[1])))[0]

        to_remove = [loc for loc in locs if loc != keeper]
        k_sp, k_jpg, k_txt = keeper

        action_move = k_sp != target
        if action_move:
            dest_dir = root / target
            if args.dry_run:
                new_stem = f"{next_stem[target]:08d}"
            else:
                new_stem = allocate_stem(target)
            dest_jpg = dest_dir / f"{new_stem}.jpg"
            dest_txt = dest_dir / f"{new_stem}.txt"
            lines.append(
                f"MOVE hash {h[:16]}... {k_sp}/{k_jpg.name} -> {target}/{dest_jpg.name} "
                f"(remove {len(locs)} pairs total incl. source)"
            )
            if not args.dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(k_jpg, dest_jpg)
                shutil.copy2(k_txt, dest_txt)
                for _, jp, tp in locs:
                    jp.unlink(missing_ok=True)
                    tp.unlink(missing_ok=True)
            n_move += 1
            n_delete_pairs += len(locs)
        else:
            lines.append(
                f"KEEP hash {h[:16]}... {target}/{k_jpg.name} (drop {len(to_remove)} other pairs)"
            )
            if not args.dry_run:
                for _, jp, tp in to_remove:
                    jp.unlink(missing_ok=True)
                    tp.unlink(missing_ok=True)
            n_delete_pairs += len(to_remove)

        n_groups_fixed += 1

    del_files = n_delete_pairs * 2
    summary = [
        "",
        "=== dedupe_laion_sample_by_hash ===",
        f"laion_root={root}",
        f"dry_run={args.dry_run}",
        f"split_priority=train > train_calib > val > test",
        f"duplicate_hash_groups={len(multi):,}",
        f"groups_resolved={n_groups_fixed:,}",
        (
            f"pairs_removed={n_delete_pairs:,} (file_deletes={del_files:,})"
            if not args.dry_run
            else f"pairs_would_remove={n_delete_pairs:,} (file_deletes_would_be={del_files:,})"
        ),
        f"moves_into_target_split={n_move:,}",
        f"cross_split_groups_sample (up to 30):",
    ]
    summary.extend("  " + x for x in cross_split_examples[:30])
    if len(cross_split_examples) > 30:
        summary.append(f"  ... and {len(cross_split_examples) - 30} more")
    summary.append("")
    summary.append("--- per-hash actions (truncated to 2000 lines) ---")
    summary.extend(lines[:2000])
    if len(lines) > 2000:
        summary.append(f"... truncated {len(lines) - 2000} more MOVE/KEEP lines")

    for s in summary[:15]:
        logger.info(s)
    if len(summary) > 15:
        logger.info("... (full detail in report file)")

    text = "\n".join(summary) + "\n"
    report_path.write_text(text, encoding="utf-8")
    logger.info("Wrote report: %s", report_path)

    if args.dry_run:
        logger.info("Dry-run: no files changed. Re-run without --dry-run to apply.")
        sys.exit(0)


if __name__ == "__main__":
    main()
