#!/usr/bin/env python3
"""
Build margin-localization figures and sanity tables from per-query .npz dumps
produced by noop_certificate_speedup.py --cert-per-query-dir.

Writes (in --out-dir):
  - margin_localization_grid.png / .pdf   — one row per (backbone,direction), one col per method
  - margin_localization_aggregate.png / .pdf — mean curves over cells, one panel per method
  - margin_localization_buckets.md — per cell×method bucket table (n, cert%, change%, gains, harms, net, cumul ΔR@1)

.npz layout (see noop_certificate_speedup.save_cert_per_query_npz):
  margin, certified, base_top1, refined_top1, gt

Usage:
  python experiments/margin_localization_report.py \\
      --npz-dir evaluation_results/tables_GPU/cert_per_query_coco \\
      --out-dir evaluation_results/tables_GPU/margin_localization
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from psi_slack_pkg.noop_certificate_speedup import MARGIN_BUCKET_DEFS  # noqa: E402

METHOD_ORDER = ["CSLS", "QB_Norm", "DB_Norm", "NNN"]
METHOD_LABEL = {"CSLS": "CSLS", "QB_Norm": "QB-Norm", "DB_Norm": "DB-Norm", "NNN": "NNN"}


def parse_npz_name(path: Path) -> tuple[str, str, str] | None:
    stem = path.stem
    parts = stem.split("__")
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def bucket_table_from_npz(
    margin: np.ndarray,
    certified: np.ndarray,
    base_top1: np.ndarray,
    refined_top1: np.ndarray,
    gt: np.ndarray,
) -> tuple[list[dict], int]:
    valid = gt >= 0
    margin = margin[valid].astype(np.float64)
    certified = certified[valid].astype(np.float64)
    base_top1 = base_top1[valid]
    refined_top1 = refined_top1[valid]
    gt = gt[valid]
    n_gt = int(len(margin))
    if n_gt == 0:
        rows = [
            {
                "bucket": lab,
                "n": 0,
                "cert_pct": 0.0,
                "change_pct": 0.0,
                "gains": 0,
                "harms": 0,
                "net": 0,
                "delta_R1_slice": 0.0,
                "cumul_delta_R1": 0.0,
            }
            for _, _, lab in MARGIN_BUCKET_DEFS
        ]
        return rows, 0

    base_ok = base_top1 == gt
    ref_ok = refined_top1 == gt
    rank_changed = base_top1 != refined_top1
    contrib = ref_ok.astype(np.float64) - base_ok.astype(np.float64)

    cumul = 0.0
    rows: list[dict] = []
    for lo, hi, lab in MARGIN_BUCKET_DEFS:
        sub = (margin >= lo) & (margin < hi)
        nb = int(sub.sum())
        if nb > 0:
            cert_p = 100.0 * float(certified[sub].mean())
            chg_p = 100.0 * float(rank_changed[sub].mean())
            gains = int((~base_ok[sub] & ref_ok[sub]).sum())
            harms = int((base_ok[sub] & ~ref_ok[sub]).sum())
            net = gains - harms
            slice_d = float(contrib[sub].sum()) / float(n_gt)
        else:
            cert_p = chg_p = 0.0
            gains = harms = net = 0
            slice_d = 0.0
        cumul += slice_d
        rows.append(
            {
                "bucket": lab,
                "n": nb,
                "cert_pct": cert_p,
                "change_pct": chg_p,
                "gains": gains,
                "harms": harms,
                "net": net,
                "delta_R1_slice": slice_d,
                "cumul_delta_R1": cumul,
            }
        )
    return rows, n_gt


def _plot_triple(ax, ax_r, rows: list[dict], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("pip install matplotlib") from e

    x = np.arange(len(rows))
    y1 = [r["cert_pct"] for r in rows]
    y2 = [r["change_pct"] for r in rows]
    y3 = [r["cumul_delta_R1"] for r in rows]
    labels = [r["bucket"] for r in rows]

    ax.plot(x, y1, "o-", color="C0", label="% certified")
    ax.plot(x, y2, "s-", color="C1", label="% rank-changed")
    ax.set_ylim(0, 100)
    ax.set_ylabel("% (GT queries)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax_r.plot(x, y3, "^-", color="C2", linewidth=2, label="cumul. ΔR@1 (pp)")
    ax_r.axhline(0.0, color="gray", linestyle=":", linewidth=0.7)
    ax_r.set_ylabel("cumul. ΔR@1 vs cos (pp)")
    ax.set_title(title, fontsize=9)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)


def discover_npz_index(npz_dir: Path) -> dict[tuple[str, str], dict[str, Path]]:
    """(backbone, direction) -> {method_stem: path}"""
    out: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    for p in sorted(npz_dir.glob("*.npz")):
        parsed = parse_npz_name(p)
        if parsed is None:
            continue
        bb, dire, meth = parsed
        out[(bb, dire)][meth] = p
    return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=project_root / "evaluation_results/tables_GPU/margin_localization")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as e:
        raise ImportError("pip install matplotlib") from e

    npz_dir = Path(args.npz_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = discover_npz_index(npz_dir)
    if not index:
        raise SystemExit(f"No *.npz found under {npz_dir}")

    cells = sorted(index.keys(), key=lambda t: (t[0], t[1]))
    nrows = len(cells)
    ncols = len(METHOD_ORDER)

    # ---------- grid PNG/PDF ----------
    fig_w = 3.6 * ncols
    fig_h = max(2.8, 2.4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True)
    fig.suptitle("Margin localization (global no-op cert): certified / rank-changed / cumul. ΔR@1", fontsize=11)

    md_lines = [
        "# Margin localization — per-bucket sanity",
        "",
        f"Generated: `{datetime.now().isoformat()}`",
        f"Source npz dir: `{npz_dir.resolve()}`",
        "",
        "Columns: **n** queries (GT) in bucket; **cert%** = global certificate rate; **chg%** = refined ≠ cosine top-1; "
        "**gains** / **harms** / **net** = refined fix / refined break vs cosine; **Δ slice** = R@1 mass in bucket (pp); "
        "**cumul** = cumulative ΔR@1 vs cosine.",
        "",
    ]

    # Collect for aggregate: method -> list of (n_buckets,) arrays
    series_per_method: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)

    for ri, (bb, dd) in enumerate(cells):
        md_lines.append(f"## {bb} | {dd}")
        md_lines.append("")
        for ci, meth in enumerate(METHOD_ORDER):
            ax = axes[ri][ci]
            ax_r = ax.twinx()
            path = index[(bb, dd)].get(meth)
            if path is None:
                ax.set_title(f"{METHOD_LABEL[meth]} — missing npz")
                continue
            z = np.load(path)
            rows, n_gt = bucket_table_from_npz(
                z["margin"],
                z["certified"],
                z["base_top1"],
                z["refined_top1"],
                z["gt"],
            )
            title = f"{bb} {dd} | {METHOD_LABEL[meth]} (n_GT={n_gt})"
            _plot_triple(ax, ax_r, rows, title)
            y1 = np.array([r["cert_pct"] for r in rows], dtype=np.float64)
            y2 = np.array([r["change_pct"] for r in rows], dtype=np.float64)
            y3 = np.array([r["cumul_delta_R1"] for r in rows], dtype=np.float64)
            series_per_method[meth].append((y1, y2, y3))

            md_lines.append(f"### {METHOD_LABEL[meth]}")
            md_lines.append("")
            md_lines.append("| bucket | n | cert% | chg% | gains | harms | net | ΔR@1 slice | cumul ΔR@1 |")
            md_lines.append("|---|--:|---:|---:|---:|---:|---:|---:|---:|")
            for r in rows:
                md_lines.append(
                    f"| {r['bucket']} | {r['n']} | {r['cert_pct']:.2f} | {r['change_pct']:.2f} | "
                    f"{r['gains']} | {r['harms']} | {r['net']:+d} | {r['delta_R1_slice']:+.5f} | {r['cumul_delta_R1']:+.5f} |"
                )
            md_lines.append("")

    grid_png = out_dir / "margin_localization_grid.png"
    grid_pdf = out_dir / "margin_localization_grid.pdf"
    fig.savefig(grid_png, dpi=args.dpi)
    fig.savefig(grid_pdf)
    plt.close(fig)
    print(f"Wrote {grid_png}\n  {grid_pdf}", flush=True)

    # ---------- aggregate ----------
    fig_a, axes_a = plt.subplots(1, ncols, figsize=(3.6 * ncols, 3.8), squeeze=False, constrained_layout=True)
    fig_a.suptitle("Aggregate over backbone×direction cells (mean ± 1 std)", fontsize=11)
    x = np.arange(len(MARGIN_BUCKET_DEFS))
    labels = [t[2] for t in MARGIN_BUCKET_DEFS]
    for ci, meth in enumerate(METHOD_ORDER):
        ax = axes_a[0][ci]
        ax_r = ax.twinx()
        lst = series_per_method.get(meth, [])
        if not lst:
            ax.set_title(f"{METHOD_LABEL[meth]} — no data")
            continue
        Y1 = np.stack([t[0] for t in lst], axis=0)
        Y2 = np.stack([t[1] for t in lst], axis=0)
        Y3 = np.stack([t[2] for t in lst], axis=0)
        m1, s1 = Y1.mean(0), Y1.std(0)
        m2, s2 = Y2.mean(0), Y2.std(0)
        m3, s3 = Y3.mean(0), Y3.std(0)
        ax.plot(x, m1, "o-", color="C0", label="mean % certified")
        ax.fill_between(x, m1 - s1, m1 + s1, color="C0", alpha=0.2)
        ax.plot(x, m2, "s-", color="C1", label="mean % rank-changed")
        ax.fill_between(x, m2 - s2, m2 + s2, color="C1", alpha=0.2)
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("%")
        ax.grid(True, alpha=0.3)
        ax_r.plot(x, m3, "^-", color="C2", linewidth=2, label="mean cumul. ΔR@1")
        ax_r.fill_between(x, m3 - s3, m3 + s3, color="C2", alpha=0.2)
        ax_r.axhline(0.0, color="gray", linestyle=":", linewidth=0.7)
        ax_r.set_ylabel("cumul. ΔR@1 (pp)")
        ax.set_title(METHOD_LABEL[meth])
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_r.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)

    agg_png = out_dir / "margin_localization_aggregate.png"
    agg_pdf = out_dir / "margin_localization_aggregate.pdf"
    fig_a.savefig(agg_png, dpi=args.dpi)
    fig_a.savefig(agg_pdf)
    plt.close(fig_a)
    print(f"Wrote {agg_png}\n  {agg_pdf}", flush=True)

    md_path = out_dir / "margin_localization_buckets.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
