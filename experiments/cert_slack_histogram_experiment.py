#!/usr/bin/env python3
"""Per-query B_q and slack for no-op certificates (COCO 8×4×2).

Computes, for each backbone × direction cell (8 cells), each baseline method
(CSLS, QB-Norm, DB-Norm, NNN), and each slack mode:

  - **global**: psi_min = min_gallery psi(c)
  - **candidate-local K=50**: psi_min = min over cosine top-50 candidates

Definitions (same as psi_slack_pkg/noop_certificate_speedup.py):

  - margin m(q) = s(q,c1*) - s(q,c2*) with cosine scores s
  - B_q = psi_eff(c1*) - psi_min  (certificate RHS)
  - slack = m - B_q ; certified iff slack >= 0

Outputs:

  1. ``CertSlackHistogram_coco_all.json`` — histograms + stats for every cell /
     method / mode (B_q and slack).
  2. ``cert_slack_per_query/<bb>__<dir>__<MethodStem>__slack_<mode>.npz``
     — 64 NPZ files with arrays:
     query_id, margin, B_q, slack, certified, changed_full,
     cosine_top1, full_rerank_top1.

Usage:
  CUDA_VISIBLE_DEVICES=0 python experiments/cert_slack_histogram_experiment.py \\
    --dataset coco_captions \\
    --json-out evaluation_results/tables_GPU/CertSlackHistogram_coco_all.json \\
    --npz-dir evaluation_results/tables_GPU/cert_slack_per_query
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from psi_slack_pkg.embeddings import (  # noqa: E402
    build_gt_mapping,
    load_embeddings,
)
from psi_slack_pkg.noop_certificate_speedup import (  # noqa: E402
    METHOD_NPZ_STEM,
    _build_psi,
    _cos_top1_and_margin,
    _full_top1,
    compute_Bq_and_slack,
)

DEFAULT_BACKBONES = ("clip", "siglip", "blip", "eva_clip_l14")
DEFAULT_DIRECTIONS = ("i2t", "t2i")
METHODS = ("CSLS", "QB-Norm", "DB-Norm", "NNN")

SLACK_MODES: Tuple[Tuple[str, int], ...] = (
    ("global", 0),
    ("localK50", 50),
)


def _histogram_stats(
    x: np.ndarray, n_bins: int, include_slack_frac: bool = False
) -> Dict[str, Any]:
    x = x[np.isfinite(x)].astype(np.float64)
    if x.size == 0:
        out: Dict[str, Any] = {
            "bin_edges": [],
            "counts": [],
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "n": 0,
        }
        if include_slack_frac:
            out["frac_slack_positive"] = float("nan")
            out["frac_slack_nonneg"] = float("nan")
        return out
    counts, edges = np.histogram(x, bins=n_bins)
    out = {
        "bin_edges": edges.tolist(),
        "counts": counts.tolist(),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "n": int(x.size),
    }
    if include_slack_frac:
        out["frac_slack_positive"] = float(np.mean(x > 0.0))
        out["frac_slack_nonneg"] = float(np.mean(x >= 0.0))
    return out


def _dedupe_i2t_queries(
    q_emb: torch.Tensor, qids: np.ndarray
) -> Tuple[torch.Tensor, np.ndarray]:
    if q_emb.shape[0] != len(qids):
        raise ValueError("query_emb / query_ids length mismatch")
    _, uix = np.unique(qids, return_index=True)
    uix.sort()
    return q_emb[uix], qids[uix]


def run_one_cell(
    dataset: str,
    backbone: str,
    direction: str,
    device: torch.device,
    csls_k: int,
    qb_tau: float,
    db_tau: float,
    nnn_k: int,
    nnn_w: float,
    npz_dir: Path,
    n_bins: int,
    cos_topk_cap: int,
) -> Dict[str, Any]:
    q_emb, g_emb, qids, gids = load_embeddings(dataset, direction, backbone)
    if direction == "i2t":
        q_emb, qids = _dedupe_i2t_queries(q_emb, qids)

    q_emb = F.normalize(q_emb.to(device), dim=1)
    g_emb = F.normalize(g_emb.to(device), dim=1)
    qids_list = [str(x) for x in qids]

    sims = q_emb @ g_emb.T
    cos_top1, margin = _cos_top1_and_margin(sims)
    m_gallery = sims.shape[1]
    k_shortlist = min(int(cos_topk_cap), m_gallery)
    cos_topk_idx = sims.topk(k_shortlist, dim=1).indices

    kw = dict(
        csls_k=csls_k,
        qb_tau=qb_tau,
        db_tau=db_tau,
        nnn_k=nnn_k,
        nnn_w=nnn_w,
    )

    cell_payload: Dict[str, Any] = {
        "cell": f"{backbone}|{direction}",
        "n_queries": int(sims.shape[0]),
        "n_gallery": int(m_gallery),
        "cos_topk_cap": int(k_shortlist),
        "methods": [],
    }

    for method in METHODS:
        psi = _build_psi(method, sims, **kw)
        full_top1 = _full_top1(method, sims, psi, **kw)
        method_row: Dict[str, Any] = {"method": method, "slack_modes": {}}

        for mode_tag, cert_k in SLACK_MODES:
            B_q_t, slack_t = compute_Bq_and_slack(
                psi, cos_top1, margin, cos_topk_idx, cert_k
            )
            certified_t = slack_t >= 0.0
            changed_t = full_top1 != cos_top1

            margin_np = margin.detach().cpu().numpy().astype(np.float32)
            B_q_np = B_q_t.detach().cpu().numpy().astype(np.float32)
            slack_np = slack_t.detach().cpu().numpy().astype(np.float32)
            cert_np = certified_t.detach().cpu().numpy().astype(np.bool_)
            chg_np = changed_t.detach().cpu().numpy().astype(np.bool_)
            c1_np = cos_top1.detach().cpu().numpy().astype(np.int64)
            f1_np = full_top1.detach().cpu().numpy().astype(np.int64)

            stem = METHOD_NPZ_STEM[method]
            fn = npz_dir / f"{backbone}__{direction}__{stem}__slack_{mode_tag}.npz"
            fn.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                fn,
                query_id=np.asarray(qids_list, dtype=object),
                margin=margin_np,
                B_q=B_q_np,
                slack=slack_np,
                certified=cert_np,
                changed_full=chg_np,
                cosine_top1=c1_np,
                full_rerank_top1=f1_np,
            )

            method_row["slack_modes"][mode_tag] = {
                "cert_local_topk": int(cert_k),
                "npz": str(fn.relative_to(project_root)),
                "B_q_histogram": _histogram_stats(B_q_np, n_bins, False),
                "slack_histogram": _histogram_stats(
                    slack_np, n_bins, include_slack_frac=True
                ),
            }

        cell_payload["methods"].append(method_row)

    return cell_payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="coco_captions")
    ap.add_argument(
        "--backbones",
        type=str,
        default=",".join(DEFAULT_BACKBONES),
        help="Comma-separated backbone ids (folder embeddings_<name>).",
    )
    ap.add_argument(
        "--directions",
        type=str,
        default=",".join(DEFAULT_DIRECTIONS),
    )
    ap.add_argument("--csls-k", type=int, default=20)
    ap.add_argument("--qb-tau", type=float, default=20.0)
    ap.add_argument("--db-tau", type=float, default=20.0)
    ap.add_argument("--nnn-k", type=int, default=64)
    ap.add_argument("--nnn-w", type=float, default=0.5)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--n-bins", type=int, default=80)
    ap.add_argument(
        "--cos-topk-cap",
        type=int,
        default=50,
        help="Shortlist size for cosine top-K (must be >= local slack K).",
    )
    ap.add_argument(
        "--json-out",
        type=str,
        default="evaluation_results/tables_GPU/CertSlackHistogram_coco_all.json",
    )
    ap.add_argument(
        "--npz-dir",
        type=str,
        default="evaluation_results/tables_GPU/cert_slack_per_query",
    )
    args = ap.parse_args()

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    backbones = [x.strip() for x in args.backbones.split(",") if x.strip()]
    directions = [x.strip() for x in args.directions.split(",") if x.strip()]
    npz_dir = project_root / args.npz_dir
    json_path = project_root / args.json_out

    if args.cos_topk_cap < 50:
        raise SystemExit("--cos-topk-cap must be >= 50 for localK50 mode.")

    cfg = {
        "dataset": args.dataset,
        "backbones": backbones,
        "directions": directions,
        "csls_k": args.csls_k,
        "qb_tau": args.qb_tau,
        "db_tau": args.db_tau,
        "nnn_k": args.nnn_k,
        "nnn_w": args.nnn_w,
        "n_bins": args.n_bins,
        "cos_topk_cap": args.cos_topk_cap,
        "slack_modes": [{"name": n, "cert_local_topk": k} for n, k in SLACK_MODES],
        "methods": list(METHODS),
        "generated": datetime.now().isoformat(),
        "definitions": {
            "margin": "cosine s(q,c1*) - s(q,c2*)",
            "B_q": "psi_eff(c1*) - psi_min (global min_psi or min over cos-top-K)",
            "slack": "margin - B_q ; certified iff slack >= 0",
            "changed_full": "full_rerank_top1 != cosine_top1",
            "psi_eff": "See noop_certificate_speedup.py docstring per method.",
        },
    }

    cells: List[Dict[str, Any]] = []
    for bb in backbones:
        for direc in directions:
            tag = f"{bb}|{direc}"
            print(f"[run] {tag}", flush=True)
            cells.append(
                run_one_cell(
                    dataset=args.dataset,
                    backbone=bb,
                    direction=direc,
                    device=device,
                    csls_k=args.csls_k,
                    qb_tau=args.qb_tau,
                    db_tau=args.db_tau,
                    nnn_k=args.nnn_k,
                    nnn_w=args.nnn_w,
                    npz_dir=npz_dir,
                    n_bins=args.n_bins,
                    cos_topk_cap=args.cos_topk_cap,
                )
            )

    payload = {
        "description": "Per-query certificate slack experiment: B_q and slack histograms "
        "for 8 cells × 4 methods × 2 slack modes (global vs cos-top-50 local min psi).",
        "config": cfg,
        "cells": cells,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}", flush=True)
    print(f"NPZs under {npz_dir}", flush=True)


if __name__ == "__main__":
    main()
