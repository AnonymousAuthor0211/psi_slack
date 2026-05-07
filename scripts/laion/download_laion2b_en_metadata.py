#!/usr/bin/env python3
"""
Download LAION-2B-en *metadata* parquet shards from Hugging Face (no images).

Dataset page (check current repo id on HF): laion/laion2B-en
Files look like: part-xxxxx-of-yyyyy.snappy.parquet

Usage:
  pip install huggingface_hub tqdm

  # List shards (no download)
  python scripts/laion/download_laion2b_en_metadata.py --list-only

  # Download first N shards (smoke test)
  python scripts/laion/download_laion2b_en_metadata.py \\
      --output-dir data/laion2b_en_meta --max-shards 4

  # Download all .parquet files (very large — ensure disk and ToS compliance)
  python scripts/laion/download_laion2b_en_metadata.py --output-dir data/laion2b_en_meta
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError as e:
    raise SystemExit(
        "Install huggingface_hub: pip install huggingface_hub tqdm\n" + str(e)
    ) from e


def _parquet_filenames(repo_id: str, revision: str | None) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
    out = [f for f in files if f.endswith(".parquet")]
    # Prefer metadata-style names if mixed
    out.sort()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Download LAION-2B-en parquet metadata from Hugging Face")
    p.add_argument("--repo-id", type=str, default="laion/laion2B-en", help="HF dataset repo id")
    p.add_argument("--revision", type=str, default=None, help="Optional branch / commit hash")
    p.add_argument("--output-dir", type=Path, default=Path("data/laion2b_en_meta"))
    p.add_argument("--max-shards", type=int, default=None, help="Download at most this many parquet files")
    p.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional regex; only filenames matching are downloaded (e.g. 'part-0000[0-1].*')",
    )
    p.add_argument("--list-only", action="store_true", help="List parquet filenames and exit")
    args = p.parse_args()

    names = _parquet_filenames(args.repo_id, args.revision)
    if args.pattern:
        rx = re.compile(args.pattern)
        names = [n for n in names if rx.search(n)]
    if args.max_shards is not None:
        names = names[: args.max_shards]

    logger.info("Found %d parquet file(s) to consider", len(names))
    if args.list_only:
        for n in names[:50]:
            print(n)
        if len(names) > 50:
            print(f"... and {len(names) - 50} more")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i, remote_name in enumerate(names):
        dest = args.output_dir / Path(remote_name).name
        if dest.exists() and dest.stat().st_size > 0:
            logger.info("[%d/%d] skip existing %s", i + 1, len(names), dest.name)
            continue
        logger.info("[%d/%d] downloading %s", i + 1, len(names), remote_name)
        path = hf_hub_download(
            repo_id=args.repo_id,
            filename=remote_name,
            repo_type="dataset",
            revision=args.revision,
            local_dir=str(args.output_dir),
            local_dir_use_symlinks=False,
        )
        logger.info("  -> %s", path)

    logger.info("Done. Parquet files under %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
