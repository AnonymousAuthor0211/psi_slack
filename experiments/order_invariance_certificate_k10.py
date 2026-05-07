#!/usr/bin/env python3
"""Top-K order-invariance / set-invariance certificates — K sweep (default 2,5,10).

Implements (per query, per method ψ):

  **Cosine shortlist** — ``topk(K+1)`` for m_K = s(c_(K)) − s(c_(K+1)); top-K for gaps.

  **Set-invariance (Theorem 2)** — two RHS variants:

  - *global*: ``m_K >= max ψ(top-K) − min ψ(gallery)`` (original; very conservative at larger K).
  - *local / candidate-local (remark)*: ``m_K >= max ψ(top-K) − min ψ(top-K)`` (= ``Psi_K``).

  **Order-invariance**

  - *loose (Theorem 3)*: ``set_fires ∧ (m_K_pair >= Psi_K)`` with ``m_K_pair`` =
    min consecutive cosine gap within top-K.
  - *exact (Proposition 5)*: ``set_fires ∧ ∀i<K : (s_i−s_{i+1}) >= (ψ(c_(i))−ψ(c_(i+1)))``.

  Reranked top-K uses the same objective as ``certificate_recall_at_k`` (CSLS/QB/NNN: ``s−ψ``;
  DB-Norm: ``log s + τ(s−ψ)``).

  **Metrics** — MRR@K / nDCG@K vs cosine top-K vs reranker top-K using ``build_gt_mapping``
  (multi-positive when applicable). Aggregates labeled ``*_cos_vs_rerank`` are *not* comparable to
  Top1Certificate gated-policy drift; sanity checks use ``*_on_order_exact_*``.

Outputs (defaults):

  - ``evaluation_results/tables_GPU/OrderInvariance_coco.json`` (aggregate over K sweep)
  - ``evaluation_results/tables_GPU/order_invariance_per_query/*.npz``
    (one file per backbone × direction × method × **effective K**; with default ``--K 2,5,10``
    this is 3×32 = **96** NPZs unless some K is clamped).

Usage::

  CUDA_VISIBLE_DEVICES=0 python experiments/order_invariance_certificate_k10.py \\
      --dataset coco_captions --K 2,5,10 \\
      --json-out evaluation_results/tables_GPU/OrderInvariance_coco.json

Single-K legacy (only K=10, 32 NPZs)::

  python experiments/order_invariance_certificate_k10.py --K 10 --npz-dir ...
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from psi_slack_pkg.embeddings import build_gt_mapping, load_embeddings  # noqa: E402
from psi_slack_pkg.noop_certificate_speedup import METHOD_NPZ_STEM, _build_psi  # noqa: E402

METHODS = ("CSLS", "QB-Norm", "DB-Norm", "NNN")
DEFAULT_BACKBONES = ("clip", "siglip", "blip", "eva_clip_l14")
DEFAULT_DIRECTIONS = ("i2t", "t2i")
DEFAULT_K_LIST = "2,5,10"
FLOAT_TOL = 1e-7


def _try_load_groundtruth(
    dataset: str,
    backbone: str,
    direction: str,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
) -> Optional[Dict[str, List[int]]]:
    try:
        mod = importlib.import_module("scripts.load_groundtruth")
        fn = getattr(mod, "load_groundtruth", None)
        if callable(fn):
            return fn(
                dataset=dataset,
                backbone=backbone,
                direction=direction,
                query_ids=query_ids,
                gallery_ids=gallery_ids,
            )
    except (ImportError, Exception):
        pass
    return None


def _method_scores(method: str, sims: torch.Tensor, psi: torch.Tensor, **kw: Any) -> torch.Tensor:
    if method == "DB-Norm":
        eps = 1e-12
        tau = float(kw["db_tau"])
        return torch.log(sims.clamp_min(eps)) + tau * (sims - psi.unsqueeze(0))
    return sims - psi.unsqueeze(0)


def _dedupe_i2t_queries(
    q_emb: torch.Tensor, qids: np.ndarray
) -> Tuple[torch.Tensor, np.ndarray]:
    _, uix = np.unique(qids, return_index=True)
    uix.sort()
    return q_emb[uix], qids[uix]


def _per_query_mrr_ndcg_k(
    top_k_ids: np.ndarray,
    gt_set: Set[int],
    k: int,
) -> Tuple[float, float]:
    hit_rank: Optional[int] = None
    for pos in range(min(len(top_k_ids), k)):
        if int(top_k_ids[pos]) in gt_set:
            hit_rank = pos + 1
            break
    mrr = (1.0 / hit_rank) if hit_rank is not None else 0.0

    rel = [1 if int(top_k_ids[j]) in gt_set else 0 for j in range(min(len(top_k_ids), k))]
    dcg = sum(rel[j] / math.log2(j + 2) for j in range(len(rel)))
    n_rel = len(gt_set)
    ideal_n = min(n_rel, k)
    idcg = sum(1.0 / math.log2(j + 2) for j in range(ideal_n))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0
    return mrr, ndcg


def run_cell_method(
    sims: torch.Tensor,
    psi: torch.Tensor,
    method: str,
    kw: Dict[str, Any],
    gt_map: Dict[str, List[int]],
    qids_list: List[str],
    has_gt: np.ndarray,
    K_eff: int,
    cos_vals_k: torch.Tensor,
    cos_idx_k: torch.Tensor,
    topkp1_vals: torch.Tensor,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Cosine top-(K_eff+1) and top-K_eff slices already computed (shared across methods)."""
    n_q = sims.shape[0]
    m_K = topkp1_vals[:, K_eff - 1] - topkp1_vals[:, K_eff]

    topk_vals = cos_vals_k
    consec_gaps = topk_vals[:, :-1] - topk_vals[:, 1:]
    m_K_pair = consec_gaps.min(dim=1).values

    topk_idx = cos_idx_k.long()
    psi_flat = psi.reshape(-1)
    psi_topk = psi_flat[topk_idx]
    psi_max_topk = psi_topk.max(dim=1).values
    psi_min_topk = psi_topk.min(dim=1).values
    Psi_K = psi_max_topk - psi_min_topk

    psi_min_global = psi_flat.min()
    set_rhs_global = psi_max_topk - psi_min_global
    set_rhs_local = psi_max_topk - psi_min_topk

    tol = FLOAT_TOL
    set_fires_global = (m_K + tol) >= set_rhs_global
    set_fires_local = (m_K + tol) >= set_rhs_local

    order_loose_global = set_fires_global & ((m_K_pair + tol) >= Psi_K)
    order_loose_local = set_fires_local & ((m_K_pair + tol) >= Psi_K)

    psi_consec_diff = psi_topk[:, :-1] - psi_topk[:, 1:]
    per_pair_ok = (consec_gaps + tol) >= psi_consec_diff
    per_pair_all_ok = per_pair_ok.all(dim=1)

    order_exact_global = set_fires_global & per_pair_all_ok
    order_exact_local = set_fires_local & per_pair_all_ok

    scores_r = _method_scores(method, sims, psi, **kw)
    _, rerank_topk_idx = scores_r.topk(K_eff, dim=1)

    idx_cos_np = topk_idx.detach().cpu().numpy().astype(np.int64)
    idx_r_np = rerank_topk_idx.detach().cpu().numpy().astype(np.int64)

    mrr_c = np.zeros(n_q, dtype=np.float64)
    mrr_r = np.zeros(n_q, dtype=np.float64)
    nd_c = np.zeros(n_q, dtype=np.float64)
    nd_r = np.zeros(n_q, dtype=np.float64)
    drift_mrr = np.zeros(n_q, dtype=np.float64)
    drift_ndcg = np.zeros(n_q, dtype=np.float64)

    k_ev = K_eff
    for i in range(n_q):
        if not has_gt[i]:
            mrr_c[i] = mrr_r[i] = nd_c[i] = nd_r[i] = math.nan
            drift_mrr[i] = drift_ndcg[i] = math.nan
            continue
        gts = gt_map.get(qids_list[i], [])
        if not gts:
            mrr_c[i] = mrr_r[i] = nd_c[i] = nd_r[i] = math.nan
            drift_mrr[i] = drift_ndcg[i] = math.nan
            continue
        gset = set(int(g) for g in gts)
        mc, nc = _per_query_mrr_ndcg_k(idx_cos_np[i], gset, k_ev)
        mr, nr = _per_query_mrr_ndcg_k(idx_r_np[i], gset, k_ev)
        mrr_c[i], nd_c[i], mrr_r[i], nd_r[i] = mc, nc, mr, nr
        drift_mrr[i] = abs(mc - mr)
        drift_ndcg[i] = abs(nc - nr)

    valid = has_gt & np.array([bool(gt_map.get(q)) for q in qids_list], dtype=bool)
    drift_m = drift_mrr[np.isfinite(drift_mrr)]
    drift_n = drift_ndcg[np.isfinite(drift_ndcg)]

    sg_np = set_fires_global.detach().cpu().numpy().astype(np.bool_)
    sl_np = set_fires_local.detach().cpu().numpy().astype(np.bool_)
    olg_np = order_loose_global.detach().cpu().numpy().astype(np.bool_)
    oll_np = order_loose_local.detach().cpu().numpy().astype(np.bool_)
    oeg_np = order_exact_global.detach().cpu().numpy().astype(np.bool_)
    oel_np = order_exact_local.detach().cpu().numpy().astype(np.bool_)

    def _safe_max(mask: np.ndarray) -> float:
        return float(np.max(np.abs(drift_mrr[mask]))) if mask.any() else 0.0

    def _safe_max_ndcg(mask: np.ndarray) -> float:
        return float(np.max(np.abs(drift_ndcg[mask]))) if mask.any() else 0.0

    summary: Dict[str, Any] = {
        "method": method,
        "K": int(K_eff),
        "n_queries": int(n_q),
        "n_eval_metrics": int(valid.sum()),
        "fire_rate_set_global": float(sg_np.mean()) if n_q else 0.0,
        "fire_rate_set_local": float(sl_np.mean()) if n_q else 0.0,
        "n_set_global": int(sg_np.sum()),
        "n_set_local": int(sl_np.sum()),
        "fire_rate_order_loose_global": float(olg_np.mean()) if n_q else 0.0,
        "fire_rate_order_loose_local": float(oll_np.mean()) if n_q else 0.0,
        "fire_rate_order_exact_global": float(oeg_np.mean()) if n_q else 0.0,
        "fire_rate_order_exact_local": float(oel_np.mean()) if n_q else 0.0,
        "n_order_loose_global": int(olg_np.sum()),
        "n_order_loose_local": int(oll_np.sum()),
        "n_order_exact_global": int(oeg_np.sum()),
        "n_order_exact_local": int(oel_np.sum()),
        "max_abs_MRR_drift_on_order_exact_global": _safe_max(oeg_np),
        "max_abs_nDCG_drift_on_order_exact_global": _safe_max_ndcg(oeg_np),
        "max_abs_MRR_drift_on_order_exact_local": _safe_max(oel_np),
        "max_abs_nDCG_drift_on_order_exact_local": _safe_max_ndcg(oel_np),
        "mean_abs_MRR_drift_cos_vs_rerank": (
            float(np.mean(np.abs(drift_m))) if drift_m.size else float("nan")
        ),
        "mean_abs_nDCG_drift_cos_vs_rerank": (
            float(np.mean(np.abs(drift_n))) if drift_n.size else float("nan")
        ),
        "max_abs_MRR_drift_cos_vs_rerank": (
            float(np.max(np.abs(drift_m))) if drift_m.size else float("nan")
        ),
        "max_abs_nDCG_drift_cos_vs_rerank": (
            float(np.max(np.abs(drift_n))) if drift_n.size else float("nan")
        ),
        # Legacy aliases (global loose order / global set)
        "fire_rate_set_invariance": float(sg_np.mean()) if n_q else 0.0,
        "fire_rate_order_invariance": float(olg_np.mean()) if n_q else 0.0,
        "n_order_fires": int(olg_np.sum()),
        "max_abs_MRR_drift_on_order_fires": _safe_max(olg_np),
        "max_abs_nDCG_drift_on_order_fires": _safe_max_ndcg(olg_np),
    }

    arrays: Dict[str, np.ndarray] = {
        "query_id": np.asarray(qids_list, dtype=object),
        "m_K": m_K.detach().cpu().numpy().astype(np.float32),
        "m_K_pair": m_K_pair.detach().cpu().numpy().astype(np.float32),
        "Psi_K": Psi_K.detach().cpu().numpy().astype(np.float32),
        "set_rhs_global": set_rhs_global.detach().cpu().numpy().astype(np.float32),
        "set_rhs_local": set_rhs_local.detach().cpu().numpy().astype(np.float32),
        "set_fires_global": sg_np,
        "set_fires_local": sl_np,
        "order_loose_fires_global": olg_np,
        "order_loose_fires_local": oll_np,
        "order_exact_fires_global": oeg_np,
        "order_exact_fires_local": oel_np,
        "per_pair_ok": per_pair_ok.detach().cpu().numpy().astype(np.bool_),
        "mrr_cosine_K": mrr_c.astype(np.float32),
        "mrr_rerank_K": mrr_r.astype(np.float32),
        "ndcg_cosine_K": nd_c.astype(np.float32),
        "ndcg_rerank_K": nd_r.astype(np.float32),
        "abs_MRR_drift": drift_mrr.astype(np.float32),
        "abs_nDCG_drift": drift_ndcg.astype(np.float32),
        "cosine_topK_indices": idx_cos_np,
        "rerank_topK_indices": idx_r_np,
    }
    return summary, arrays


def run_cell(
    dataset: str,
    backbone: str,
    direction: str,
    device: torch.device,
    npz_dir: Path,
    K_request: int,
    csls_k: int,
    qb_tau: float,
    db_tau: float,
    nnn_k: int,
    nnn_w: float,
) -> Dict[str, Any]:
    q_emb, g_emb, qids, gids = load_embeddings(dataset, direction, backbone)
    if direction == "i2t":
        q_emb, qids = _dedupe_i2t_queries(q_emb, qids)

    q_emb = F.normalize(q_emb.to(device), dim=1)
    g_emb = F.normalize(g_emb.to(device), dim=1)
    qids_list = [str(x) for x in qids]

    gt_map_alt = _try_load_groundtruth(dataset, backbone, direction, qids, gids)
    gt_map = (
        gt_map_alt
        if gt_map_alt is not None
        else build_gt_mapping(qids, gids, direction)
    )
    has_gt = np.array([bool(gt_map.get(q)) for q in qids_list], dtype=bool)

    sims = q_emb @ g_emb.T
    n_g = sims.shape[1]
    K_eff = min(int(K_request), max(n_g - 1, 1))
    if K_eff < 2:
        raise ValueError(f"Need K>=2 and gallery>=K+1; got K_req={K_request}, n_g={n_g}, K_eff={K_eff}")
    if n_g < K_eff + 1:
        raise ValueError(f"Gallery too small: need >= K+1={K_eff+1} items, got {n_g}")

    topkp1_vals, topkp1_idx = sims.topk(K_eff + 1, dim=1)
    cos_vals_k = topkp1_vals[:, :K_eff]
    cos_idx_k = topkp1_idx[:, :K_eff]

    kw = dict(
        csls_k=csls_k,
        qb_tau=qb_tau,
        db_tau=db_tau,
        nnn_k=nnn_k,
        nnn_w=nnn_w,
    )

    methods_out: List[Dict[str, Any]] = []
    npz_dir.mkdir(parents=True, exist_ok=True)

    for method in METHODS:
        psi = _build_psi(method, sims, **kw)
        summary, arrs = run_cell_method(
            sims,
            psi,
            method,
            kw,
            gt_map,
            qids_list,
            has_gt,
            K_eff,
            cos_vals_k,
            cos_idx_k,
            topkp1_vals,
        )
        stem = METHOD_NPZ_STEM[method]
        fn = npz_dir / f"{backbone}__{direction}__{stem}__orderK{K_eff}.npz"
        np.savez_compressed(fn, **arrs)
        summary["per_query_npz"] = str(fn.relative_to(project_root))
        summary["ground_truth_source"] = (
            "load_groundtruth" if gt_map_alt is not None else "build_gt_mapping"
        )
        summary["definitions"] = {
            "m_K": "s(q,c_(K)) - s(q,c_(K+1))",
            "m_K_pair": "min_{1<=i<K} (s(c_(i)) - s(c_(i+1))) on cosine top-K",
            "Psi_K": "max ψ(top-K) - min ψ(top-K)",
            "set_global": "m_K >= max ψ(top-K) - min ψ(gallery)",
            "set_local": "m_K >= max ψ(top-K) - min ψ(top-K) (= Psi_K); candidate-local remark",
            "order_loose": "set_fires ∧ (m_K_pair >= Psi_K)",
            "order_exact": "set_fires ∧ ∀ consecutive pairs: Δs >= Δψ (Proposition 5)",
            "FLOAT_TOL": FLOAT_TOL,
        }
        methods_out.append(summary)

    return {
        "cell": f"{backbone}|{direction}",
        "K": int(K_eff),
        "K_requested": int(K_request),
        "n_queries": int(len(qids_list)),
        "n_queries_with_gt": int(has_gt.sum()),
        "n_gallery": int(g_emb.shape[0]),
        "ground_truth_source": (
            "load_groundtruth" if gt_map_alt is not None else "build_gt_mapping"
        ),
        "methods": methods_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=str, default="coco_captions")
    ap.add_argument("--backbones", type=str, default=",".join(DEFAULT_BACKBONES))
    ap.add_argument("--directions", type=str, default=",".join(DEFAULT_DIRECTIONS))
    ap.add_argument(
        "--K",
        type=str,
        default=DEFAULT_K_LIST,
        help="Comma-separated K values (one cell block per K×backbone×direction).",
    )
    ap.add_argument("--csls-k", type=int, default=20)
    ap.add_argument("--qb-tau", type=float, default=20.0)
    ap.add_argument("--db-tau", type=float, default=20.0)
    ap.add_argument("--nnn-k", type=int, default=64)
    ap.add_argument("--nnn-w", type=float, default=0.5)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument(
        "--json-out",
        "--out-json",
        dest="json_out",
        type=str,
        default="evaluation_results/tables_GPU/OrderInvariance_coco.json",
        help="Aggregate JSON path.",
    )
    ap.add_argument(
        "--npz-dir",
        "--per-query-dir",
        dest="npz_dir",
        type=str,
        default="evaluation_results/tables_GPU/order_invariance_per_query",
    )
    args = ap.parse_args()

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    backbones = [x.strip() for x in args.backbones.split(",") if x.strip()]
    directions = [x.strip() for x in args.directions.split(",") if x.strip()]
    ks = [int(x.strip()) for x in args.K.split(",") if x.strip()]
    npz_dir = project_root / args.npz_dir
    json_path = project_root / args.json_out

    cfg = {
        "dataset": args.dataset,
        "backbones": backbones,
        "directions": directions,
        "K_list": ks,
        "csls_k": args.csls_k,
        "qb_tau": args.qb_tau,
        "db_tau": args.db_tau,
        "nnn_k": args.nnn_k,
        "nnn_w": args.nnn_w,
        "methods": list(METHODS),
        "generated": datetime.now().isoformat(),
        "notes": {
            "fire_rates": "Prefer fire_rate_order_exact_* and fire_rate_set_local for interpretability at larger K.",
            "drift_cos_vs_rerank": "Not comparable to Top1Certificate_effect_R5R10 gated drift.",
        },
    }

    cells: List[Dict[str, Any]] = []
    for K in ks:
        for bb in backbones:
            for direc in directions:
                print(f"[order-invariance] K={K} {bb}|{direc}", flush=True)
                cells.append(
                    run_cell(
                        dataset=args.dataset,
                        backbone=bb,
                        direction=direc,
                        device=device,
                        npz_dir=npz_dir,
                        K_request=K,
                        csls_k=args.csls_k,
                        qb_tau=args.qb_tau,
                        db_tau=args.db_tau,
                        nnn_k=args.nnn_k,
                        nnn_w=args.nnn_w,
                    )
                )

    payload = {
        "description": "Top-K order/set certificate firing rates; K sweep; global vs local set; loose vs exact order.",
        "config": cfg,
        "cells": cells,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
