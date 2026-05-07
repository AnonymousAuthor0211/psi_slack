#!/usr/bin/env python3
"""
Experiment A — Learned gates vs ψ-slack at matched skip rates.

Trains / calibrates three gates on an 80/20 stratified split (label: cosine top-1
matches full rerank top-1), tunes thresholds on TRAIN to hit target skip rates,
then reports TEST disagreement between ψ-slack (local K=50) and each gate.

Features (per query): margin m; cosine ranks 2–5 and consecutive gaps; B_q global,
B_q local (K=50); ψ(c1*), min/max/mean/std ψ over cos-top-K; entropy of softmax over
cos-top-K; rank of argmin ψ in shortlist.

Train/test indices are **shared across methods** in each cell (stratified by whether cosine top-1
matches **CSLS** full-gallery rerank top-1).

Fourth gate (default): **split-calibrated uncertainty** u = 1 − p(c₁|q) on the cosine shortlist;
τ is the target-skip quantile on a held-out **calibration slice of train** (then applied to train/test).
Legacy: ``--legacy-conformal-aps`` restores APS singleton (often degenerate on peaked shortlists).

**Skip workload:** By default each fold skips exactly ``k = round(s * n_fold)`` queries (deterministic
tie-break), so **test skip rate matches** ``s`` up to floating display. Use ``--no-exact-skip-per-fold``
for train-quantile thresholds (legacy; test skip only approximate).

Cells: omit ``--cells`` or pass ``--cells all`` to run **all** ``--backbones`` × ``--directions`` (default:
four backbones × i2t/t2i for COCO).

Added **full-gallery TEST diagnostics** per method: R@1 cosine vs R@1 full rerank (queries with GT).

Outputs: paper_draft/tables_info/Table_LearnedGates_MatchedSkip.json

Depends: numpy, torch, sklearn; optional xgboost for the XGBoost baseline
(HistGradientBoostingClassifier is used if xgboost is absent).

Example:
  CUDA_VISIBLE_DEVICES=0 python experiments/learned_gates_matched_skip.py \\
    --dataset coco_captions --device 0 --cells clip:i2t,siglip:t2i
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

METHODS = ("CSLS", "QB-Norm", "DB-Norm", "NNN")
SKIP_RATES = (0.15, 0.25, 0.35, 0.45, 0.55)
DEFAULT_BACKBONES = ("clip", "siglip", "blip", "eva_clip_l14")
DEFAULT_DIRECTIONS = ("i2t", "t2i")

# Column order in `build_feature_matrix` (for NPZ sanity checks).
FEATURE_COLUMNS = (
    "margin",
    "cos_rank2",
    "cos_rank3",
    "cos_rank4",
    "cos_rank5",
    "gap_s2_s3",
    "gap_s3_s4",
    "gap_s4_s5",
    "B_q_global",
    "B_q_local",
    "psi_c1",
    "psi_min_topk",
    "psi_max_topk",
    "psi_mean_topk",
    "psi_std_topk",
    "entropy_softmax_topk",
    "rank_argmin_psi_topk",
)


def _dedupe_i2t_queries(
    q_emb: torch.Tensor, qids: np.ndarray
) -> Tuple[torch.Tensor, np.ndarray]:
    if q_emb.shape[0] != len(qids):
        raise ValueError("query_emb / query_ids length mismatch")
    _, uix = np.unique(qids, return_index=True)
    uix.sort()
    return q_emb[uix], qids[uix]


def _maybe_load_npz_Bslack(
    npz_dir: Path | None,
    backbone: str,
    direction: str,
    method: str,
) -> Dict[str, np.ndarray] | None:
    """Optional merge of precomputed B_q/slack from cert_slack_histogram NPZs."""
    if npz_dir is None or not npz_dir.is_dir():
        return None
    stem = METHOD_NPZ_STEM[method]
    fg = npz_dir / f"{backbone}__{direction}__{stem}__slack_global.npz"
    fl = npz_dir / f"{backbone}__{direction}__{stem}__slack_localK50.npz"
    if not fg.is_file() or not fl.is_file():
        return None
    dg = np.load(fg, allow_pickle=True)
    dl = np.load(fl, allow_pickle=True)
    qg = [str(x) for x in dg["query_id"]]
    ql = [str(x) for x in dl["query_id"]]
    if qg != ql:
        return None
    return {
        "query_id": np.asarray(qg, dtype=object),
        "B_q_global": dg["B_q"].astype(np.float32),
        "slack_global": dg["slack"].astype(np.float32),
        "B_q_local": dl["B_q"].astype(np.float32),
        "slack_local_npz": dl["slack"].astype(np.float32),
    }


def build_feature_matrix(
    sims: torch.Tensor,
    psi: torch.Tensor,
    cos_top1: torch.Tensor,
    margin: torch.Tensor,
    cos_topk_idx: torch.Tensor,
    cos_topk_vals: torch.Tensor,
    cert_k: int,
    temperature: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns X [N,F], slack_local (ψ-slack with local min over cos-top-K)."""
    device = sims.device
    n, k_cap = cos_topk_vals.shape
    k_eff = min(cert_k, k_cap)

    B_glob, _ = compute_Bq_and_slack(
        psi, cos_top1, margin, cos_topk_idx=None, cert_local_topk=0
    )
    B_loc, slack_local = compute_Bq_and_slack(
        psi, cos_top1, margin, cos_topk_idx, cert_local_topk=k_eff
    )

    top5_v, _ = sims.topk(min(5, sims.shape[1]), dim=1)
    pad = min(5, sims.shape[1])
    s2 = top5_v[:, 1] if pad > 1 else torch.zeros(n, device=device)
    s3 = top5_v[:, 2] if pad > 2 else s2
    s4 = top5_v[:, 3] if pad > 3 else s3
    s5 = top5_v[:, 4] if pad > 4 else s4

    idx_k = cos_topk_idx[:, :k_eff].long()
    psi_topk = psi[idx_k]
    psi_c1 = psi[cos_top1]
    psi_min_k = psi_topk.min(dim=1).values
    psi_max_k = psi_topk.max(dim=1).values
    psi_mean_k = psi_topk.mean(dim=1)
    psi_std_k = psi_topk.std(dim=1, unbiased=False)
    rank_argmin = psi_topk.argmin(dim=1).to(torch.float32)

    logits = cos_topk_vals[:, :k_eff] / float(temperature)
    p = torch.softmax(logits, dim=1)
    log_p = torch.log(p.clamp_min(1e-12))
    entropy = -(p * log_p).sum(dim=1)

    feats = torch.stack(
        [
            margin,
            s2,
            s3,
            s4,
            s5,
            (s2 - s3),
            (s3 - s4),
            (s4 - s5),
            B_glob,
            B_loc,
            psi_c1,
            psi_min_k,
            psi_max_k,
            psi_mean_k,
            psi_std_k,
            entropy,
            rank_argmin,
        ],
        dim=1,
    )

    return (
        feats.detach().cpu().numpy().astype(np.float32),
        slack_local.detach().cpu().numpy().astype(np.float32),
    )


def _full_top1_np(method: str, sims: torch.Tensor, psi: torch.Tensor, kw: dict) -> np.ndarray:
    return _full_top1(method, sims, psi, **kw).detach().cpu().numpy().astype(np.int64)


def skip_from_quantile_high(train_score: np.ndarray, test_score: np.ndarray, s: float):
    """Skip queries with score >= quantile(train, 1-s)."""
    thr = np.quantile(train_score, 1.0 - s)
    return train_score >= thr, test_score >= thr


def skip_psi_slack_high(train_slack: np.ndarray, test_slack: np.ndarray, s: float):
    """Skip when slack is among the highest fraction s (easy / redundant rerank)."""
    thr = np.quantile(train_slack, 1.0 - s)
    return train_slack >= thr, test_slack >= thr


def skip_mask_top_k_fraction(score: np.ndarray, s: float, *, skip_largest: bool) -> np.ndarray:
    """
    Exactly ``k = round(s * n)`` skipped queries on this fold (deterministic tie-break: ascending index).
    If skip_largest=True, skip the k largest scores (ψ-slack, high confidence).
    If skip_largest=False, skip the k smallest scores (low uncertainty u = 1−p(c₁)).
    """
    x = np.asarray(score, dtype=np.float64).reshape(-1)
    n = int(x.shape[0])
    k = min(n, max(0, int(round(float(s) * n))))
    mask = np.zeros(n, dtype=bool)
    if k == 0:
        return mask
    secondary = np.arange(n)
    if skip_largest:
        order = np.lexsort((secondary, -x))
    else:
        order = np.lexsort((secondary, x))
    mask[order[:k]] = True
    return mask


def train_xgb_or_hgb(X_tr: np.ndarray, y_tr: np.ndarray):
    if np.unique(y_tr).size < 2:
        from sklearn.dummy import DummyClassifier

        clf = DummyClassifier(strategy="prior")
        clf.fit(X_tr, y_tr)
        return clf, "dummy_prior_single_class_subset"

    try:
        import xgboost as xgb  # type: ignore

        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=8,
            eval_metric="logloss",
        )
        clf.fit(X_tr, y_tr)
        return clf, "xgboost"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=250,
            random_state=42,
        )
        clf.fit(X_tr, y_tr)
        return clf, "sklearn_hist_gbrt"


def predict_proba_pos(clf, X: np.ndarray) -> np.ndarray:
    proba = clf.predict_proba(X)
    if proba.shape[1] == 1:
        return proba[:, 0]
    return proba[:, 1]


def aps_set_sizes(probs: np.ndarray, alpha: float) -> np.ndarray:
    """Prediction set size: smallest |S| with mass >= 1−α over softmax shortlist probs."""
    pv = np.sort(probs, axis=1)[:, ::-1]
    cum = np.cumsum(pv, axis=1)
    thr = 1.0 - float(alpha)
    n, k = probs.shape
    out = np.zeros(n, dtype=np.int32)
    for i in range(n):
        j = int(np.searchsorted(cum[i], thr, side="left"))
        out[i] = min(j + 1, k)
    return out


def calibrate_aps_alpha_singleton_rate(p_calib: np.ndarray, s: float) -> float:
    """Legacy APS: pick α so singleton fraction ≈ s (often degenerate on peaked shortlists)."""
    alphas = np.linspace(1e-4, 0.99, 200)
    best_a = 0.5
    best_err = 1.0
    for a in alphas:
        sz = aps_set_sizes(p_calib, float(a))
        frac = float(np.mean(sz == 1))
        err = abs(frac - s)
        if err < best_err:
            best_err = err
            best_a = float(a)
    return best_a


def skip_masks_split_calib_uncertainty(
    pc_tr: np.ndarray,
    pc_te: np.ndarray,
    pc_calib: np.ndarray,
    s: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Split-calibrated gate: nonconformity u = 1 − p(c₁|q) on cosine shortlist softmax.
    Threshold τ is the s-quantile of u on the calibration slice only, then applied to all TRAIN
    and TEST queries (distribution-free selective baseline).
    """
    u_calib = 1.0 - pc_calib
    tau = float(np.quantile(u_calib, s))
    u_tr = 1.0 - pc_tr
    u_te = 1.0 - pc_te
    return u_tr <= tau, u_te <= tau, tau


def skip_exact_metrics(mask: np.ndarray) -> Dict[str, Any]:
    m = np.asarray(mask, dtype=bool).reshape(-1)
    k = int(m.sum())
    n = int(m.size)
    return {"numerator_skipped": k, "denominator": n, "rate": float(k / n) if n else float("nan")}


def gated_top1(skip: np.ndarray, cos_top1: np.ndarray, full_top1: np.ndarray) -> np.ndarray:
    return np.where(skip, cos_top1, full_top1)


def pct_disagree_vs_full(skip: np.ndarray, cos_top1: np.ndarray, full_top1: np.ndarray) -> float:
    """% test queries where gated top-1 ≠ full-gallery rerank top-1."""
    g = gated_top1(skip, cos_top1, full_top1)
    return float(100.0 * np.mean(g != full_top1))


def r1_hits_top1(top1: np.ndarray, gt_map: Dict[str, List[int]], qids_row: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (hit, has_gt) booleans aligned with top1 rows.
    hit is False when has_gt is False (ignored in means).
    """
    hit = np.zeros(len(top1), dtype=bool)
    has_gt = np.zeros(len(top1), dtype=bool)
    for i, qid in enumerate(qids_row):
        gts = gt_map.get(str(qid), [])
        if not gts:
            continue
        has_gt[i] = True
        hit[i] = int(top1[i]) in set(gts)
    return hit, has_gt


def r1_rate_on_gt(hit: np.ndarray, has_gt: np.ndarray) -> float | None:
    sel = has_gt
    if not np.any(sel):
        return None
    return float(hit[sel].mean())


def r1_gap_percentage_pts(gated_top: np.ndarray, full_top: np.ndarray, gt_map: Dict[str, List[int]], qids_row: List[str]) -> float | None:
    """R@1(gated) − R@1(full), percentage points, restricted to queries with GT."""
    hg_f, mask = r1_hits_top1(full_top, gt_map, qids_row)
    if not np.any(mask):
        return None
    hg_g, _ = r1_hits_top1(gated_top, gt_map, qids_row)
    return float(100.0 * (hg_g[mask].mean() - hg_f[mask].mean()))


def run_one_method(
    method: str,
    sims: torch.Tensor,
    kw: dict,
    cos_topk_idx: torch.Tensor,
    cos_topk_vals: torch.Tensor,
    cos_top1: torch.Tensor,
    margin: torch.Tensor,
    temperature: float,
    skip_rates: Tuple[float, ...],
    seed: int,
    optional_npz: Dict[str, np.ndarray] | None,
    idx_tr: np.ndarray,
    idx_te: np.ndarray,
    qids: np.ndarray,
    gt_map: Dict[str, List[int]],
    legacy_conformal_aps: bool,
    exact_skip_per_fold: bool,
) -> Dict[str, Any]:
    psi = _build_psi(method, sims, **kw)
    cos_top1_np = cos_top1.detach().cpu().numpy().astype(np.int64)
    full_top1_np = _full_top1_np(method, sims, psi, kw)
    y = (cos_top1_np == full_top1_np).astype(np.int32)

    X_t, slack_loc_t = build_feature_matrix(
        sims,
        psi,
        cos_top1,
        margin,
        cos_topk_idx,
        cos_topk_vals,
        cert_k=50,
        temperature=temperature,
    )

    verify_note: Dict[str, Any] = {}
    if optional_npz is not None:
        verify_note["max_abs_diff_slack_local_vs_npz"] = float(
            np.max(np.abs(slack_loc_t - optional_npz["slack_local_npz"]))
        )
        i_bg = FEATURE_COLUMNS.index("B_q_global")
        i_bl = FEATURE_COLUMNS.index("B_q_local")
        verify_note["max_abs_diff_Bq_global_vs_npz"] = float(
            np.max(np.abs(X_t[:, i_bg] - optional_npz["B_q_global"]))
        )
        verify_note["max_abs_diff_Bq_local_vs_npz"] = float(
            np.max(np.abs(X_t[:, i_bl] - optional_npz["B_q_local"]))
        )

    logits_full = cos_topk_vals / float(temperature)
    p_full = torch.softmax(logits_full, dim=1).detach().cpu().numpy().astype(np.float64)

    from sklearn.model_selection import train_test_split

    X_tr, X_te = X_t[idx_tr], X_t[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    sl_tr, sl_te = slack_loc_t[idx_tr], slack_loc_t[idx_te]
    c1_tr, c1_te = cos_top1_np[idx_tr], cos_top1_np[idx_te]
    f1_tr, f1_te = full_top1_np[idx_tr], full_top1_np[idx_te]
    pc_tr, pc_te = p_full[idx_tr, 0], p_full[idx_te, 0]
    p_calib_tr = p_full[idx_tr]

    idx_positions_tr = np.arange(len(idx_tr))
    try:
        idx_fit_rel, idx_calib_rel = train_test_split(
            idx_positions_tr,
            test_size=0.25,
            stratify=y_tr,
            random_state=seed + 1,
        )
    except ValueError:
        idx_fit_rel, idx_calib_rel = train_test_split(
            idx_positions_tr,
            test_size=0.25,
            stratify=None,
            random_state=seed + 1,
        )
    idx_fit_rows = idx_tr[idx_fit_rel]
    idx_calib_rows = idx_tr[idx_calib_rel]

    clf, clf_name = train_xgb_or_hgb(X_t[idx_fit_rows], y[idx_fit_rows])
    proba_tr = predict_proba_pos(clf, X_tr)
    proba_te = predict_proba_pos(clf, X_te)

    qids_te = [str(qids[i]) for i in idx_te]
    hit_cos_te, has_gt_te = r1_hits_top1(c1_te, gt_map, qids_te)
    hit_full_te, has_gt_te2 = r1_hits_top1(f1_te, gt_map, qids_te)
    assert np.array_equal(has_gt_te, has_gt_te2)

    full_gallery_test_diag: Dict[str, Any] = {
        "definition": "TEST fold only; R@1 among queries with ≥1 GT gallery index (full-gallery cosine / full-gallery rerank).",
        "split_anchor_note": "Train/test indices fixed per cell via CSLS agreement stratification (same fold for all methods).",
        "n_queries_test": int(len(idx_te)),
        "n_queries_with_gt_test": int(has_gt_te.sum()),
        "R_at_1_cosine_full_gallery": r1_rate_on_gt(hit_cos_te, has_gt_te),
        "R_at_1_full_rerank_this_method": r1_rate_on_gt(hit_full_te, has_gt_te),
    }

    rows: List[Dict[str, Any]] = []
    for s in skip_rates:
        if exact_skip_per_fold:
            sk_psi_tr = skip_mask_top_k_fraction(sl_tr, s, skip_largest=True)
            sk_psi_te = skip_mask_top_k_fraction(sl_te, s, skip_largest=True)
            sk_xgb_tr = skip_mask_top_k_fraction(proba_tr, s, skip_largest=True)
            sk_xgb_te = skip_mask_top_k_fraction(proba_te, s, skip_largest=True)
            sk_soft_tr = skip_mask_top_k_fraction(pc_tr, s, skip_largest=True)
            sk_soft_te = skip_mask_top_k_fraction(pc_te, s, skip_largest=True)
            tau_split: float | None = None
            alpha_hat: float | None = None
            if legacy_conformal_aps:
                alpha_hat = calibrate_aps_alpha_singleton_rate(p_full[idx_calib_rows], s)
                sz_tr_full = aps_set_sizes(p_calib_tr, alpha_hat)
                sz_te_full = aps_set_sizes(p_full[idx_te], alpha_hat)
                conf_skip_tr = sz_tr_full == 1
                conf_skip_te = sz_te_full == 1
            else:
                u_tr = 1.0 - pc_tr
                u_te = 1.0 - pc_te
                conf_skip_tr = skip_mask_top_k_fraction(u_tr, s, skip_largest=False)
                conf_skip_te = skip_mask_top_k_fraction(u_te, s, skip_largest=False)
                k_te = min(len(u_te), max(0, int(round(float(s) * len(u_te)))))
                if k_te > 0:
                    tau_split = float(np.sort(u_te)[k_te - 1])
                else:
                    tau_split = float("nan")
        else:
            sk_psi_tr, sk_psi_te = skip_psi_slack_high(sl_tr, sl_te, s)

            sk_xgb_tr, sk_xgb_te = skip_from_quantile_high(proba_tr, proba_te, s)

            sk_soft_tr, sk_soft_te = skip_from_quantile_high(pc_tr, pc_te, s)

            tau_split = None
            alpha_hat = None
            if legacy_conformal_aps:
                alpha_hat = calibrate_aps_alpha_singleton_rate(p_full[idx_calib_rows], s)
                sz_tr_full = aps_set_sizes(p_calib_tr, alpha_hat)
                sz_te_full = aps_set_sizes(p_full[idx_te], alpha_hat)
                conf_skip_tr = sz_tr_full == 1
                conf_skip_te = sz_te_full == 1
            else:
                pc_calib_slice = p_full[idx_calib_rows, 0]
                conf_skip_tr, conf_skip_te, tau_split = skip_masks_split_calib_uncertainty(
                    pc_tr,
                    pc_te,
                    pc_calib_slice,
                    s,
                )

        def disagreement(sk_a: np.ndarray, sk_b: np.ndarray) -> float:
            return float(np.mean(sk_a != sk_b))

        g_psi_te = gated_top1(sk_psi_te, c1_te, f1_te)
        g_xgb_te = gated_top1(sk_xgb_te, c1_te, f1_te)
        g_soft_te = gated_top1(sk_soft_te, c1_te, f1_te)
        g_conf_te = gated_top1(conf_skip_te, c1_te, f1_te)

        gd_test_psi_xgb = float(np.mean(g_psi_te != g_xgb_te))
        gd_test_psi_soft = float(np.mean(g_psi_te != g_soft_te))
        gd_test_psi_conf = float(np.mean(g_psi_te != g_conf_te))

        ex_test_psi = skip_exact_metrics(sk_psi_te)
        ex_test_xgb = skip_exact_metrics(sk_xgb_te)
        ex_test_soft = skip_exact_metrics(sk_soft_te)
        ex_test_conf = skip_exact_metrics(conf_skip_te)

        r1_gap_psi = r1_gap_percentage_pts(g_psi_te, f1_te, gt_map, qids_te)
        r1_gap_xgb = r1_gap_percentage_pts(g_xgb_te, f1_te, gt_map, qids_te)
        r1_gap_soft = r1_gap_percentage_pts(g_soft_te, f1_te, gt_map, qids_te)
        r1_gap_conf = r1_gap_percentage_pts(g_conf_te, f1_te, gt_map, qids_te)

        row = {
            "skip_rate_target": s,
            "skip_matching_rule": (
                "exact_top_k_per_fold_round_sn"
                if exact_skip_per_fold
                else "train_quantile_applied_to_train_and_test"
            ),
            # Flat columns (table-friendly)
            "psi_disagree_vs_full": pct_disagree_vs_full(sk_psi_te, c1_te, f1_te),
            "xgboost_disagree_vs_full": pct_disagree_vs_full(sk_xgb_te, c1_te, f1_te),
            "softmax_disagree_vs_full": pct_disagree_vs_full(sk_soft_te, c1_te, f1_te),
            "r1_gap_to_full_percentage_pts_psi_slack": r1_gap_psi,
            "r1_gap_to_full_percentage_pts_xgboost": r1_gap_xgb,
            "r1_gap_to_full_percentage_pts_softmax_pc1": r1_gap_soft,
            "achieved_skip_test_exact_psi_slack_local": ex_test_psi,
            "achieved_skip_test_exact_xgboost": ex_test_xgb,
            "achieved_skip_test_exact_softmax_pc1": ex_test_soft,
            "achieved_skip_rate_train": {
                "psi_slack_local": float(np.mean(sk_psi_tr)),
                "xgboost": float(np.mean(sk_xgb_tr)),
                "softmax_pc1": float(np.mean(sk_soft_tr)),
                **({"conformal_aps_singleton": float(np.mean(conf_skip_tr))} if legacy_conformal_aps else {"split_calib_uncertainty_1mpc1": float(np.mean(conf_skip_tr))}),
            },
            "achieved_skip_rate_test": {
                "psi_slack_local": ex_test_psi["rate"],
                "xgboost": ex_test_xgb["rate"],
                "softmax_pc1": ex_test_soft["rate"],
                **({"conformal_aps_singleton": ex_test_conf["rate"]} if legacy_conformal_aps else {"split_calib_uncertainty_1mpc1": ex_test_conf["rate"]}),
            },
            "decision_disagreement_vs_psi_TEST": {
                "xgboost_pct": 100.0 * disagreement(sk_psi_te, sk_xgb_te),
                "softmax_pc1_pct": 100.0 * disagreement(sk_psi_te, sk_soft_te),
                **({"conformal_aps_singleton_pct": 100.0 * disagreement(sk_psi_te, conf_skip_te)} if legacy_conformal_aps else {"split_calib_uncertainty_1mpc1_pct": 100.0 * disagreement(sk_psi_te, conf_skip_te)}),
            },
            "gated_top1_disagreement_vs_psi_TEST": {
                "xgboost_pct": 100.0 * gd_test_psi_xgb,
                "softmax_pc1_pct": 100.0 * gd_test_psi_soft,
                **({"conformal_aps_singleton_pct": 100.0 * gd_test_psi_conf} if legacy_conformal_aps else {"split_calib_uncertainty_1mpc1_pct": 100.0 * gd_test_psi_conf}),
            },
        }
        if legacy_conformal_aps:
            row["aps_singleton_disagree_vs_full"] = pct_disagree_vs_full(conf_skip_te, c1_te, f1_te)
            row["r1_gap_to_full_percentage_pts_aps_singleton"] = r1_gap_conf
            row["achieved_skip_test_exact_aps_singleton"] = ex_test_conf
            row["conformal_alpha_calibrated"] = alpha_hat
        else:
            row["split_calib_disagree_vs_full"] = pct_disagree_vs_full(conf_skip_te, c1_te, f1_te)
            row["r1_gap_to_full_percentage_pts_split_calib"] = r1_gap_conf
            row["achieved_skip_test_exact_split_calib"] = ex_test_conf
            if exact_skip_per_fold and tau_split is not None:
                row["split_calib_u_max_among_skipped_test"] = tau_split
            elif not exact_skip_per_fold:
                row["split_calib_threshold_tau_on_uncertainty_1mpc1"] = tau_split

        rows.append(row)

    out: Dict[str, Any] = {
        "method": method,
        "feature_columns": list(FEATURE_COLUMNS),
        "label_def": "y=1 iff cosine_top1_idx == full_rerank_top1_idx",
        "psi_slack_mode": "local cosine-top-50 min ψ",
        "skip_matching": (
            "Each fold: skip exactly k=round(s*n) queries by score rank (deterministic ties)."
            if exact_skip_per_fold
            else "Train-quantile thresholds applied to train and test (test skip rate approximate)."
        ),
        "conformal_gate": (
            "APS singleton on shortlist softmax (legacy; often degenerate)"
            if legacy_conformal_aps
            else (
                "split-calib on u=1−p(c₁): exact k smallest-u queries skipped per fold (matched workload)"
                if exact_skip_per_fold
                else "split-calibrated uncertainty u=1−p(c₁): τ = quantile_calib(u, target_skip); skip if u≤τ"
            )
        ),
        "classifier_backend": clf_name,
        "full_gallery_test_diag": full_gallery_test_diag,
        "optional_npz_verification": verify_note,
        "skip_rate_rows": rows,
    }
    return out


def parse_cells(spec: str | None, backbones: List[str], directions: List[str]) -> List[Tuple[str, str]]:
    if spec:
        s0 = spec.strip()
        if s0.lower() == "all":
            return [(bb, d) for bb in backbones for d in directions]
        pairs = []
        for part in s0.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(f"Bad --cells entry {part!r}; expected backbone:direction")
            bb, di = part.split(":", 1)
            pairs.append((bb.strip(), di.strip()))
        return pairs
    return [(bb, d) for bb in backbones for d in directions]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="coco_captions")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for gates/features.")
    ap.add_argument("--csls-k", type=int, default=20)
    ap.add_argument("--qb-tau", type=float, default=20.0)
    ap.add_argument("--db-tau", type=float, default=20.0)
    ap.add_argument("--nnn-k", type=int, default=64)
    ap.add_argument("--nnn-w", type=float, default=0.5)
    ap.add_argument("--cos-topk-cap", type=int, default=50)
    ap.add_argument(
        "--slack-npz-dir",
        type=str,
        default="evaluation_results/tables_GPU/cert_slack_per_query",
        help="If NPZs exist, sanity-check B_q/slack vs recomputation.",
    )
    ap.add_argument(
        "--cells",
        type=str,
        default="",
        help="Comma-separated backbone:direction (e.g. clip:i2t). Use `all` or leave empty for "
        "full grid: --backbones × --directions (default COCO: clip,siglip,blip,eva_clip_l14 × i2t,t2i).",
    )
    ap.add_argument("--backbones", type=str, default=",".join(DEFAULT_BACKBONES))
    ap.add_argument("--directions", type=str, default=",".join(DEFAULT_DIRECTIONS))
    ap.add_argument(
        "--legacy-conformal-aps",
        action="store_true",
        help="Use APS singleton baseline (often 0%% or 100%% skip on peaked shortlists). Default: split-calibrated u=1−p(c₁).",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for train/test and calib splits.")
    ap.set_defaults(exact_skip_per_fold=True)
    ap.add_argument(
        "--no-exact-skip-per-fold",
        dest="exact_skip_per_fold",
        action="store_false",
        help="Train-quantile thresholds (approximate skip rate on test). Default: exact k=round(s*n) skipped per fold.",
    )
    ap.add_argument(
        "--json-out",
        type=str,
        default="paper_draft/tables_info/Table_LearnedGates_MatchedSkip.json",
    )
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    backbones = [x.strip() for x in args.backbones.split(",") if x.strip()]
    directions = [x.strip() for x in args.directions.split(",") if x.strip()]
    cell_pairs = parse_cells(args.cells or None, backbones, directions)
    npz_dir = project_root / args.slack_npz_dir
    kw = dict(
        csls_k=args.csls_k,
        qb_tau=args.qb_tau,
        db_tau=args.db_tau,
        nnn_k=args.nnn_k,
        nnn_w=args.nnn_w,
    )

    payload_cells: List[Dict[str, Any]] = []
    for backbone, direction in cell_pairs:
        print(f"[cell] {backbone}|{direction}", flush=True)
        q_emb, g_emb, qids, gids = load_embeddings(args.dataset, direction, backbone)
        if direction == "i2t":
            q_emb, qids = _dedupe_i2t_queries(q_emb, qids)

        gt_map = build_gt_mapping(qids, gids, direction)

        q_emb = F.normalize(q_emb.to(device), dim=1)
        g_emb = F.normalize(g_emb.to(device), dim=1)
        sims = q_emb @ g_emb.T
        k_short = min(int(args.cos_topk_cap), sims.shape[1])
        cos_topk_vals, cos_topk_idx = sims.topk(k_short, dim=1)
        cos_top1, margin = _cos_top1_and_margin(sims)

        cos_top1_np_anchor = cos_top1.detach().cpu().numpy().astype(np.int64)
        psi_csls = _build_psi("CSLS", sims, **kw)
        full_top1_csls = _full_top1_np("CSLS", sims, psi_csls, kw)
        y_anchor = (cos_top1_np_anchor == full_top1_csls).astype(np.int32)
        idx_all = np.arange(sims.shape[0])
        from sklearn.model_selection import train_test_split as tts

        try:
            idx_tr, idx_te = tts(
                idx_all,
                test_size=0.2,
                stratify=y_anchor,
                random_state=args.seed,
            )
        except ValueError:
            idx_tr, idx_te = tts(
                idx_all,
                test_size=0.2,
                stratify=None,
                random_state=args.seed,
            )

        methods_payload = []
        for method in METHODS:
            print(f"  [method] {method}", flush=True)
            onpz = _maybe_load_npz_Bslack(npz_dir, backbone, direction, method)
            methods_payload.append(
                run_one_method(
                    method,
                    sims,
                    kw,
                    cos_topk_idx,
                    cos_topk_vals,
                    cos_top1,
                    margin,
                    args.temperature,
                    SKIP_RATES,
                    args.seed,
                    onpz,
                    idx_tr,
                    idx_te,
                    qids,
                    gt_map,
                    args.legacy_conformal_aps,
                    args.exact_skip_per_fold,
                )
            )

        payload_cells.append(
            {
                "cell": f"{backbone}|{direction}",
                "n_queries": int(sims.shape[0]),
                "n_gallery": int(sims.shape[1]),
                "full_gallery": {
                    "retrieval": "All similarities and rerank argmax are over the full gallery (size n_gallery).",
                    "train_test_split_anchor": "Stratified by whether cosine_top1 == CSLS_full_rerank_top1 (same fold for every method).",
                },
                "methods": methods_payload,
            }
        )

    out_path = project_root / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": "Learned gates vs ψ-slack (matched skip rates)",
        "generated": datetime.now().isoformat(),
        "config": {
            "dataset": args.dataset,
            "train_test_split": "80/20; stratify=y_anchor with y_anchor=(cos_top1==CSLS_full_top1); same idx_tr/idx_te for all methods",
            "fourth_gate_default": "split_calib_uncertainty_1mpc1 (τ from calib quantile on u=1−p(c₁)); use --legacy-conformal-aps for APS singleton",
            "legacy_conformal_aps": bool(args.legacy_conformal_aps),
            "exact_skip_per_fold": bool(args.exact_skip_per_fold),
            "seed": int(args.seed),
            "skip_rates": list(SKIP_RATES),
            "temperature": args.temperature,
            "cos_topk_cap": args.cos_topk_cap,
            **kw,
        },
        "cells": payload_cells,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
