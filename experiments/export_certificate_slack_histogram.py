#!/usr/bin/env python3
"""
Certificate slack histogram export (bucket summaries + optional per-query arrays).

For offset rerankers (CSLS / QB-Norm / DB-Norm / NNN), certificate uses:
    m(q) >= ψ(c1*) − ψ_min   ⇔   slack := m(q) − B_q  ≥  0
where
    B_q = ψ(c1*) − min ψ(c)
with ψ_min either **global** min_g ψ(g) or **candidate-local** min over cosine top-K.

Slack > 0  → certified no-op (sound under global rule).
Slack < 0  → not certified; refinement may change top-1.

Default artifact stores **histogram bucket summaries only** (edges + counts + stats).
Use `--per-query-dir` for optional NPZ with raw margin, B_q, slack, changed_full.

Usage:
  python experiments/export_certificate_slack_histogram.py \\
      --dataset coco_captions --backbones clip --directions i2t,t2i \\
      --out-json evaluation_results/tables_GPU/CertSlackHistogram_coco.json

  # Include candidate-local B_q (same K as cert-local-topk) + figure:
  python experiments/export_certificate_slack_histogram.py \\
      --cert-local-topk 50 --plot evaluation_results/figures_noop/CertSlack_hist.png

  # Raw per-query (large):
  python experiments/export_certificate_slack_histogram.py \\
      --per-query-dir evaluation_results/tables_GPU/cert_slack_per_query
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from psi_slack_pkg.embeddings import load_embeddings  # noqa: E402
from psi_slack_pkg.noop_certificate_speedup import (  # noqa: E402
    METHOD_NPZ_STEM,
    _build_psi,
    _cos_top1_and_margin,
    _full_top1,
)

METHODS = ("CSLS", "QB-Norm", "DB-Norm", "NNN")


def _slack_tensors(
    psi: torch.Tensor,
    cos_top1: torch.Tensor,
    margin: torch.Tensor,
    cos_topk_idx: torch.Tensor | None,
    cert_local_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (B_q, slack) per query."""
    psi_c1 = psi[cos_top1]
    if cert_local_topk >= 2 and cos_topk_idx is not None:
        k_eff = min(int(cert_local_topk), cos_topk_idx.shape[1])
        idx = cos_topk_idx[:, :k_eff]
        psi_min_q = psi[idx].min(dim=1).values
    else:
        psi_min_q = torch.full_like(psi_c1, psi.min())
    b_q = psi_c1 - psi_min_q
    slack = margin - b_q
    return b_q, slack


def _histogram_buckets(slack: np.ndarray, n_bins: int = 48) -> dict[str, Any]:
    """Linear bins from min(slack) to max(slack); all queries counted."""
    lo = float(np.min(slack))
    hi = float(np.max(slack))
    if hi <= lo:
        hi = lo + 1e-12
    counts, edges = np.histogram(slack, bins=n_bins, range=(lo, hi))
    return {
        "bin_edges": edges.astype(np.float64).tolist(),
        "counts": counts.astype(np.int64).tolist(),
        "range_used": [lo, hi],
        "n_bins": int(n_bins),
    }


def _stats(slack: np.ndarray) -> dict[str, float]:
    return {
        "n": int(slack.shape[0]),
        "mean": float(np.mean(slack)),
        "median": float(np.median(slack)),
        "std": float(np.std(slack)),
        "frac_slack_positive": float(np.mean(slack > 0)),
        "frac_slack_nonpositive": float(np.mean(slack <= 0)),
        "min": float(np.min(slack)),
        "max": float(np.max(slack)),
    }


def _run_cell(
    dataset: str,
    direction: str,
    backbone: str,
    device: torch.device,
    csls_k: int,
    qb_tau: float,
    db_tau: float,
    nnn_k: int,
    nnn_w: float,
    cos_topk_for_local: int,
    hist_bins: int,
    per_query_dir: Path | None,
    qids_list: list[str] | None = None,
) -> dict[str, Any]:
    q_emb, g_emb, qids, gids = load_embeddings(dataset, direction, backbone)
    q_emb = F.normalize(q_emb.to(device), dim=1)
    g_emb = F.normalize(g_emb.to(device), dim=1)
    qids_list = qids_list or [str(x) for x in qids]

    sims = q_emb @ g_emb.T
    cos_top1, margin = _cos_top1_and_margin(sims)
    k_available = sims.shape[1]
    k_local = min(int(cos_topk_for_local), k_available) if cos_topk_for_local >= 2 else 0
    cos_topk_idx = sims.topk(k_local, dim=1).indices if k_local >= 2 else None

    kw = {
        "csls_k": csls_k,
        "qb_tau": qb_tau,
        "db_tau": db_tau,
        "nnn_k": nnn_k,
        "nnn_w": nnn_w,
    }

    modes: dict[str, Any] = {}

    def process_mode(mode_name: str, cert_local_topk: int, ctidx: torch.Tensor | None) -> dict[str, Any]:
        out_methods: dict[str, Any] = {}
        for method in METHODS:
            psi = _build_psi(method, sims, **kw)
            full_top1 = _full_top1(method, sims, psi, **kw)
            b_q, slack = _slack_tensors(
                psi, cos_top1, margin, ctidx, cert_local_topk
            )
            slack_np = slack.detach().cpu().numpy().astype(np.float64)
            margin_np = margin.detach().cpu().numpy().astype(np.float64)
            b_np = b_q.detach().cpu().numpy().astype(np.float64)
            cos_top1_np = cos_top1.cpu().numpy().astype(np.int64)
            full_np = full_top1.cpu().numpy().astype(np.int64)
            changed_full = full_np != cos_top1_np

            hist = _histogram_buckets(slack_np, n_bins=hist_bins)
            stats = _stats(slack_np)

            row = {
                "histogram": hist,
                "stats": stats,
                "definition": {
                    "margin_m": "s(q,c1*) - s(q,c2*) cosine margin",
                    "B_q": "psi(c1*) - psi_min (same ψ as noop certificate)",
                    "slack": "m(q) - B_q  ( >0 ⇔ certified under this ψ_min rule )",
                    "changed_full": "full rerank top-1 != cosine top-1",
                },
            }

            if per_query_dir is not None:
                per_query_dir.mkdir(parents=True, exist_ok=True)
                stem = METHOD_NPZ_STEM.get(method, method.replace("-", "_"))
                fn = (
                    per_query_dir
                    / f"{backbone}__{direction}__{stem}__slack_{mode_name}.npz"
                )
                np.savez_compressed(
                    fn,
                    query_id=np.array(qids_list, dtype=object),
                    margin=margin_np.astype(np.float32),
                    B_q=b_np.astype(np.float32),
                    slack=slack_np.astype(np.float32),
                    certified=(slack_np > 0).astype(np.bool_),
                    changed_full=changed_full.astype(np.bool_),
                    cosine_top1=cos_top1_np,
                    full_rerank_top1=full_np,
                )

            out_methods[method] = row
        return out_methods

    modes["global"] = process_mode("global", 0, cos_topk_idx)

    if k_local >= 2 and cos_topk_idx is not None:
        modes[f"candidate_local_top{k_local}"] = process_mode(
            f"local{k_local}",
            k_local,
            cos_topk_idx,
        )

    return {
        "cell": f"{backbone}|{direction}",
        "n_queries": int(sims.shape[0]),
        "n_gallery": int(sims.shape[1]),
        "cosine_topk_used_for_local": k_local,
        "modes": modes,
    }


def _plot_cell(cell_data: dict, out_path: Path, dataset: str, dpi: int = 150) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib required for --plot") from e

    modes = cell_data["modes"]
    mode_key = "global"
    if mode_key not in modes:
        mode_key = next(iter(modes))
    block = modes[mode_key]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    fig.suptitle(
        f'Certificate slack $m-B_q$ ({mode_key}) — {cell_data["cell"]} | {dataset}',
        fontsize=11,
    )
    for ax, method in zip(axes.flat, METHODS):
        if method not in block:
            ax.set_visible(False)
            continue
        h = block[method]["histogram"]
        edges = np.array(h["bin_edges"])
        counts = np.array(h["counts"])
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, counts, width=np.diff(edges), align="center", color="#4C72B0", alpha=0.85)
        ax.axvline(0.0, color="#C44E52", linestyle="--", lw=1.2, label="slack=0")
        stats = block[method]["stats"]
        ax.set_title(
            f'{method}  (+:{100 * stats["frac_slack_positive"]:.1f}%)',
            fontsize=9,
        )
        ax.set_xlabel(r"$m(q) - B_q$")
        ax.set_ylabel("count")
        ax.tick_params(axis="both", labelsize=8)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", type=str, default="coco_captions")
    ap.add_argument("--backbones", type=str, default="clip")
    ap.add_argument("--directions", type=str, default="i2t,t2i")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--csls-k", type=int, default=20)
    ap.add_argument("--qb-tau", type=float, default=20.0)
    ap.add_argument("--db-tau", type=float, default=20.0)
    ap.add_argument("--nnn-k", type=int, default=64)
    ap.add_argument("--nnn-w", type=float, default=0.5)
    ap.add_argument(
        "--cert-local-topk",
        type=int,
        default=50,
        help="If >=2, also export candidate-local B_q using cosine top-K (0 = global only).",
    )
    ap.add_argument("--hist-bins", type=int, default=48)
    ap.add_argument(
        "--out-json",
        type=Path,
        default=project_root / "evaluation_results" / "tables_GPU" / "CertSlackHistogram.json",
    )
    ap.add_argument(
        "--per-query-dir",
        type=Path,
        default=None,
        help="If set, save NPZ per method/mode with margin, B_q, slack, certified, changed_full.",
    )
    ap.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Save small 2×2 histogram PNG for global slack (first backbone/direction).",
    )
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    backbones = [x.strip() for x in args.backbones.split(",") if x.strip()]
    directions = [x.strip() for x in args.directions.split(",") if x.strip()]

    payload: dict[str, Any] = {
        "description": (
            "Histogram buckets of slack = m(q) − B_q with "
            "B_q = ψ(c1*) − ψ_min (global or cosine top-K local)."
        ),
        "config": {
            "dataset": args.dataset,
            "cert_local_topk": args.cert_local_topk,
            "hist_bins": args.hist_bins,
            "csls_k": args.csls_k,
            "qb_tau": args.qb_tau,
            "db_tau": args.db_tau,
            "nnn_k": args.nnn_k,
            "nnn_w": args.nnn_w,
        },
    }

    cells = []
    first_plot_done = False
    for b in backbones:
        for d in directions:
            cell = _run_cell(
                args.dataset,
                d,
                b,
                device,
                args.csls_k,
                args.qb_tau,
                args.db_tau,
                args.nnn_k,
                args.nnn_w,
                args.cert_local_topk,
                args.hist_bins,
                Path(args.per_query_dir) if args.per_query_dir else None,
            )
            cells.append(cell)
            if args.plot and not first_plot_done:
                args.plot.parent.mkdir(parents=True, exist_ok=True)
                _plot_cell(cell, args.plot, args.dataset)
                print(f"Wrote plot {args.plot}")
                first_plot_done = True

    payload["cells"] = cells
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json}")
    if args.per_query_dir:
        print(f"Per-query NPZ under {args.per_query_dir}")


if __name__ == "__main__":
    main()
