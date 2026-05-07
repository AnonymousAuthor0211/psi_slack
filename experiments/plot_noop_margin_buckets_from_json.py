#!/usr/bin/env python3
"""
Plot margin-bucket curves from either:
  - `NoOpCertificate_<dataset>_margin_curves.json` (slim, recommended), or
  - `NoOpCertificate_<dataset>.json` (full run; also contains `margin_buckets`).

  python experiments/plot_noop_margin_buckets_from_json.py \\
      evaluation_results/tables_GPU/NoOpCertificate_coco_captions_margin_curves.json \\
      -o evaluation_results/figures_noop
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from psi_slack_pkg.noop_certificate_speedup import plot_margin_bucket_curves  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=Path)
    ap.add_argument("-o", "--out-dir", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument(
        "--one-per-method",
        action="store_true",
        help="One PNG per (cell, method) instead of a 2×2 combined figure per cell.",
    )
    args = ap.parse_args()

    data = json.loads(args.json_path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])
    cfg = data.get("config", {})

    missing = []
    for cell in cells:
        for m in cell.get("methods", []):
            if "margin_buckets" not in m:
                missing.append(f"{cell.get('cell')}/{m.get('method')}")
                break
    if missing:
        print(
            "JSON lacks `margin_buckets` (from an older run). Re-run:\n"
            "  python -m psi_slack_pkg.noop_certificate_speedup ... [--margin-plots-dir DIR]\n"
            f"Missing in: {missing[:5]}{'...' if len(missing) > 5 else ''}",
            file=sys.stderr,
        )
        sys.exit(1)

    plot_margin_bucket_curves(
        cells,
        cfg,
        args.out_dir,
        dpi=args.dpi,
        one_figure_per_method=args.one_per_method,
    )
    print(f"Figures under {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
