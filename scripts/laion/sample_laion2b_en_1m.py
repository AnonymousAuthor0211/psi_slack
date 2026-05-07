#!/usr/bin/env python3
"""
Uniform reservoir sample over parquet metadata for img2dataset / LAION-style workflows.

Supports classic LAION shards (columns URL, TEXT) and common variants (url, caption / text).
Reads shards sequentially, applies reservoir sampling on *eligible* rows only (valid URL+text,
optional English LANGUAGE filter), so the output has exactly --sample-size rows when enough
eligible rows exist across scanned files.

Usage:
  pip install pandas pyarrow tqdm

  python scripts/laion/sample_laion2b_en_1m.py \\
      --meta-dir data/laion2b_en_meta \\
      --out-csv data/laion2b_en_1m_sample.csv \\
      --sample-size 1000000 \\
      --seed 42
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

import pandas as pd
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


# Classic LAION-2B-en metadata uses URL / TEXT; img2dataset-style parquets often use url / caption.
_URL_CANDIDATES = ("URL", "url")
_TEXT_CANDIDATES = ("TEXT", "text", "caption")
# Optional metadata: (canonical_name_in_output_json, candidate source columns in parquet)
_OPTIONAL_ALIASES: List[tuple[str, tuple[str, ...]]] = [
    ("LANGUAGE", ("LANGUAGE", "language", "lang")),
    ("NSFW", ("NSFW", "nsfw", "punsafe")),
    ("similarity", ("similarity", "Similarity")),
    ("WIDTH", ("WIDTH", "width")),
    ("HEIGHT", ("HEIGHT", "height")),
]


def _schema_columns(pf: pq.ParquetFile) -> set[str]:
    return set(pf.schema_arrow.names)


def _first_present(avail: set[str], candidates: tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in avail:
            return c
    return None


class ColumnMapping:
    """Maps parquet column names to canonical URL/TEXT used internally."""

    def __init__(
        self,
        url_col: str,
        text_col: str,
        optional_sources: Dict[str, str],
        read_columns: List[str],
    ):
        self.url_col = url_col
        self.text_col = text_col
        self.optional_sources = optional_sources  # canonical -> parquet name
        self.read_columns = read_columns

    @classmethod
    def from_parquet_file(
        cls,
        pf: pq.ParquetFile,
        url_override: Optional[str],
        text_override: Optional[str],
    ) -> ColumnMapping:
        avail = _schema_columns(pf)
        url_col = url_override or _first_present(avail, _URL_CANDIDATES)
        text_col = text_override or _first_present(avail, _TEXT_CANDIDATES)
        if url_col is None or text_col is None:
            raise ValueError(
                "Could not find URL and text columns. Tried url in "
                f"{_URL_CANDIDATES} and text in {_TEXT_CANDIDATES}. "
                f"Columns present: {sorted(avail)[:60]}"
                + (" ..." if len(avail) > 60 else "")
            )
        optional_sources: Dict[str, str] = {}
        read_set = {url_col, text_col}
        for canon, candidates in _OPTIONAL_ALIASES:
            src = _first_present(avail, candidates)
            if src is not None:
                optional_sources[canon] = src
                read_set.add(src)
        read_columns = sorted(read_set)
        return cls(url_col, text_col, optional_sources, read_columns)


def _normalize_row(rec: Dict[str, Any], mapping: ColumnMapping) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "URL": rec.get(mapping.url_col),
        "TEXT": rec.get(mapping.text_col),
    }
    for canon, src in mapping.optional_sources.items():
        if src in rec:
            out[canon] = rec[src]
    return out


def _as_clean_str(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace").strip()
    s = str(x).strip()
    return s if s else None


def _row_eligible(row: Dict[str, Any], english_only: bool) -> bool:
    u = _as_clean_str(row.get("URL"))
    t = _as_clean_str(row.get("TEXT"))
    if u is None or t is None:
        return False
    if english_only and "LANGUAGE" in row and row["LANGUAGE"] is not None:
        lang = row["LANGUAGE"]
        if isinstance(lang, str) and lang.lower() != "en":
            return False
    return True


def reservoir_sample_parquets(
    parquet_files: List[str],
    sample_size: int,
    seed: int,
    english_only: bool,
    max_eligible_rows: Optional[int],
    url_col: Optional[str],
    text_col: Optional[str],
) -> pd.DataFrame:
    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    seen_eligible = 0

    for pf_path in tqdm(parquet_files, desc="parquet files"):
        pf = pq.ParquetFile(pf_path)
        mapping = ColumnMapping.from_parquet_file(pf, url_col, text_col)
        for batch_idx in range(pf.num_row_groups):
            table = pf.read_row_group(batch_idx, columns=mapping.read_columns)
            df = table.to_pandas()
            # Normalize types for CSV / img2dataset
            for c in df.columns:
                if df[c].dtype == object:
                    df[c] = df[c].where(pd.notna(df[c]), None)

            for rec in df.to_dict("records"):
                norm = _normalize_row(rec, mapping)
                if not _row_eligible(norm, english_only=english_only):
                    continue
                norm["URL"] = _as_clean_str(norm["URL"])
                norm["TEXT"] = _as_clean_str(norm["TEXT"])
                seen_eligible += 1
                if len(reservoir) < sample_size:
                    reservoir.append(norm)
                else:
                    j = rng.randint(1, seen_eligible)
                    if j <= sample_size:
                        reservoir[j - 1] = norm
                if max_eligible_rows is not None and seen_eligible >= max_eligible_rows:
                    break
            if max_eligible_rows is not None and seen_eligible >= max_eligible_rows:
                break
        if max_eligible_rows is not None and seen_eligible >= max_eligible_rows:
            break

    if not reservoir:
        raise RuntimeError("No eligible rows found; check filters and parquet columns.")
    if len(reservoir) < sample_size:
        logger.warning(
            "Reservoir has %s rows but target was %s (not enough eligible rows in scanned parquets).",
            len(reservoir),
            sample_size,
        )

    out = pd.DataFrame(reservoir)
    out = out.rename(columns={"URL": "url", "TEXT": "caption"})
    logger.info(
        "Reservoir: eligible_seen=%s reservoir_size=%s (target %s)",
        seen_eligible,
        len(out),
        sample_size,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reservoir sample LAION-2B-en metadata parquets -> CSV for img2dataset"
    )
    parser.add_argument("--meta-dir", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)
    parser.add_argument("--sample-size", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--english-only",
        action="store_true",
        help="Drop rows with LANGUAGE set and not 'en' (keeps rows with missing LANGUAGE)",
    )
    parser.add_argument(
        "--max-eligible-rows",
        type=int,
        default=None,
        help="Stop after scanning this many eligible rows (debug / partial shards)",
    )
    parser.add_argument(
        "--extra-json-column",
        type=str,
        default="laion_meta",
        help="If set, pack optional columns into one JSON column for CSV (img2dataset passes through)",
    )
    parser.add_argument(
        "--url-col",
        type=str,
        default=None,
        help="Force parquet column name for image URL (default: auto-detect URL or url)",
    )
    parser.add_argument(
        "--caption-col",
        type=str,
        default=None,
        help="Force parquet column for caption text (default: auto-detect TEXT, text, or caption)",
    )
    args = parser.parse_args()

    parquet_files = sorted(
        glob.glob(os.path.join(args.meta_dir, "*.parquet"))
        + glob.glob(os.path.join(args.meta_dir, "**", "*.parquet"), recursive=True)
    )
    # De-dupe (recursive also matches top-level on some systems)
    parquet_files = sorted(set(parquet_files))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {args.meta_dir}")

    sampled = reservoir_sample_parquets(
        parquet_files=parquet_files,
        sample_size=args.sample_size,
        seed=args.seed,
        english_only=args.english_only,
        max_eligible_rows=args.max_eligible_rows,
        url_col=args.url_col,
        text_col=args.caption_col,
    )

    # Optional: single extra column with provenance for img2dataset save_additional_columns
    extra_keys = [
        k for k in ("LANGUAGE", "NSFW", "similarity", "WIDTH", "HEIGHT") if k in sampled.columns
    ]
    if args.extra_json_column and extra_keys:
        def _pack(row) -> str:
            d = {k: row[k] for k in extra_keys if k in row.index and pd.notna(row[k])}
            return json.dumps(d, ensure_ascii=False)

        sampled[args.extra_json_column] = sampled.apply(_pack, axis=1)
        sampled = sampled.drop(columns=extra_keys, errors="ignore")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    sampled.to_csv(args.out_csv, index=False)
    logger.info("Wrote %s rows to %s", f"{len(sampled):,}", args.out_csv)


if __name__ == "__main__":
    main()
