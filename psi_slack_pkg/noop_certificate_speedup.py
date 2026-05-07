#!/usr/bin/env python3
"""
No-Op Certificate Speedup.

For each method M in {CSLS, QB-Norm, DB-Norm, NNN}:
  (1) Derive a per-query no-op certificate from per-c statistics.
  (2) Measure % certified queries.
  (3) Run gated reranking (full method only on uncertified queries; cosine top-1
      is used directly for certified queries).
  (4) Verify R@1 matches full reranking exactly (sound certificate -> 100% match).
  (5) Measure wall-clock speedup, both end-to-end (per-c stats + score+argmax)
      and rerank-only (per-c stats already cached).

------------------------------------------------------------------------------
Math: each method's argmax has the form
    argmax_c S_method(q, c) = argmax_c [s(q, c) - psi_eff(c)]
after dropping per-query constants and monotone transforms (DB-Norm has an
additional +log(s)/tau term; the certificate ignores it, see below).

  CSLS:    psi_eff(c) = r_S(c) / 2,                r_S(c) = mean_{q in NN_k(c)} s(q, c)
  QB-Norm: psi_eff(c) = log Z(c) / tau,            Z(c)   = sum_q exp(tau * s(q, c))
  DB-Norm: psi_eff(c) = log Z(c) / tau   (LOOSE; ignores +log s/tau term)
  NNN:     psi_eff(c) = w * alpha(c),              alpha(c) = mean_{q in NN_k(c)} s(q, c)

Per-query no-op certificate (sufficient):
    m(q) >= psi_eff(c1*) - min_c psi_eff(c),   where m(q) = s(q, c1*) - s(q, c2*)

Soundness:
    For any c != c1*, gap(c) := s(q, c1*) - s(q, c) >= m(q) (since c2* maximizes
    s(q, c) over c != c1*). If m(q) >= psi_eff(c1*) - psi_min_global, then
        gap(c) >= m(q) >= psi_eff(c1*) - psi_min_global >= psi_eff(c1*) - psi_eff(c),
    so s(q, c1*) - psi_eff(c1*) >= s(q, c) - psi_eff(c), and c1* is the argmax.

DB-Norm extra term: actual argmax is over [log s + tau*(s - psi_eff)]. Since
log s(q, c1*) >= log s(q, c) (cosine top-1 has the largest s), the +log(s) term
only HELPS c1*; if the loose certificate (using psi_eff = log Z/tau) holds, the
actual DB-Norm argmax remains c1*. So the certificate is sound for DB-Norm too.

Usage:
  CUDA_VISIBLE_DEVICES=4 python -m psi_slack_pkg.noop_certificate_speedup \
      --dataset coco_captions --backbones clip,siglip,blip,eva_clip_l14 \
      --directions i2t,t2i

  Candidate-local (tight surrogate) certificate — min psi over cos-top-K only:
      ... --cert-local-topk 50
  (Use 0 for global min_psi; output stem gets `_localTopK{K}` when K>=2.)

  NNN with larger neighbor pool in ψ(c) (output stem adds `_nnnK{k}` when k≠64):
      ... --nnn-k 512

  Per-query arrays for margin-localization reports (then run `experiments/margin_localization_report.py`):
      ... --cert-per-query-dir evaluation_results/tables_GPU/cert_per_query_coco
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from psi_slack_pkg._paths import REPO_ROOT as project_root
from psi_slack_pkg.embeddings import build_gt_mapping, load_embeddings  # noqa: E402


OUTPUT_DIR = project_root / "evaluation_results" / "tables_GPU"

# Cosine margin m = s(q,c1)−s(q,c2); buckets ordered low→high (ambiguous → confident)
MARGIN_BUCKET_DEFS: list[tuple[float, float, str]] = [
    (0.0, 0.01, "[0, 0.01)"),
    (0.01, 0.03, "[0.01, 0.03)"),
    (0.03, 0.05, "[0.03, 0.05)"),
    (0.05, float("inf"), "[0.05, ∞)"),
]

# Filename segment for {backbone}__{direction}__{stem}.npz (avoid "__" in stem)
METHOD_NPZ_STEM = {"CSLS": "CSLS", "QB-Norm": "QB_Norm", "DB-Norm": "DB_Norm", "NNN": "NNN"}


def _first_gt_index_array(gt_map: dict, qids_list: list[str]) -> np.ndarray:
    """One GT gallery index per query; first caption if multiple; -1 if none."""
    out = np.full(len(qids_list), -1, dtype=np.int64)
    for i, q in enumerate(qids_list):
        g = gt_map.get(q)
        if g and len(g) > 0:
            out[i] = int(g[0])
    return out


def save_cert_per_query_npz(
    path: Path,
    margin: np.ndarray,
    certified: np.ndarray,
    base_top1: np.ndarray,
    refined_top1: np.ndarray,
    gt: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        margin=margin.astype(np.float32),
        certified=certified.astype(np.bool_),
        base_top1=base_top1.astype(np.int64),
        refined_top1=refined_top1.astype(np.int64),
        gt=gt.astype(np.int64),
    )


def margin_bucket_rows(
    margin: torch.Tensor,
    certified: torch.Tensor,
    cos_top1: torch.Tensor,
    full_top1: torch.Tensor,
    cos_correct: np.ndarray,
    full_correct: np.ndarray,
    has_gt: np.ndarray,
) -> list[dict]:
    """
    Per bucket (queries with GT only):
      pct_certified: fraction where no-op certificate fires
      pct_rank_changed: fraction where full method top-1 ≠ cosine top-1
      delta_R1_in_bucket: sum_i (full_correct[i]-cos_correct[i]) / n_gt  (R@1 mass in pp)
      cumulative_delta_R1: running sum of delta_R1_in_bucket over buckets in margin order
    """
    n_gt = int(has_gt.sum())
    if n_gt == 0:
        return [
            {
                "bucket": lab,
                "n": 0,
                "pct_certified": 0.0,
                "pct_rank_changed": 0.0,
                "delta_R1_in_bucket": 0.0,
                "cumulative_delta_R1": 0.0,
            }
            for _, _, lab in MARGIN_BUCKET_DEFS
        ]

    margin_np = margin.detach().cpu().numpy()
    cert_np = certified.detach().cpu().numpy()
    c1_np = cos_top1.detach().cpu().numpy()
    f1_np = full_top1.detach().cpu().numpy()

    margin_m = margin_np[has_gt]
    cert_m = cert_np[has_gt]
    rank_changed_m = f1_np[has_gt] != c1_np[has_gt]
    contrib_m = full_correct[has_gt].astype(np.float64) - cos_correct[has_gt].astype(np.float64)

    cumulative = 0.0
    rows: list[dict] = []
    for lo, hi, label in MARGIN_BUCKET_DEFS:
        sub = (margin_m >= lo) & (margin_m < hi)
        n_b = int(sub.sum())
        if n_b > 0:
            pct_cert = 100.0 * float(cert_m[sub].mean())
            pct_rank = 100.0 * float(rank_changed_m[sub].mean())
            slice_delta = float(contrib_m[sub].sum()) / float(n_gt)
        else:
            pct_cert = 0.0
            pct_rank = 0.0
            slice_delta = 0.0
        cumulative += slice_delta
        rows.append(
            {
                "bucket": label,
                "n": n_b,
                "pct_certified": pct_cert,
                "pct_rank_changed": pct_rank,
                "delta_R1_in_bucket": slice_delta,
                "cumulative_delta_R1": cumulative,
            }
        )
    return rows


def _plot_one_method_axes(
    ax,
    ax_r,
    mrow: dict,
    labels: list[str],
    x: np.ndarray,
    y1: list[float],
    y2: list[float],
    y3: list[float],
) -> None:
    ax.plot(x, y1, "o-", color="C0", label="% certified")
    ax.plot(x, y2, "s-", color="C1", label="% rank-changed")
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of queries in bucket (GT)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.grid(True, alpha=0.3)
    ax_r.plot(x, y3, "^-", color="C2", linewidth=2, label="cumul. ΔR@1 (pp)")
    ax_r.set_ylabel("cumulative ΔR@1 vs cosine (pp)")
    ax_r.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title(f"{mrow['method']}  (n_cert globally: {mrow.get('n_certified', '—')})")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)


def plot_margin_bucket_curves(
    results: list[dict],
    cfg: dict,
    out_dir: Path,
    dpi: int = 150,
    one_figure_per_method: bool = False,
) -> None:
    """
    Default: one 2×2 figure per cell (backbone|direction).
    If one_figure_per_method: one PNG per (cell, method) — three curves each.
    Left axis: % certified, % rank-changed; right axis: cumulative ΔR@1 (pp).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Plotting requires matplotlib: pip install matplotlib") from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = cfg.get("dataset", "dataset")

    for cell in results:
        cell_tag = str(cell["cell"]).replace("|", "_").replace("/", "_")
        methods = cell["methods"]

        if one_figure_per_method:
            for mrow in methods:
                mb = mrow.get("margin_buckets")
                if not mb:
                    continue
                fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
                ax_r = ax.twinx()
                x = np.arange(len(mb))
                y1 = [float(b["pct_certified"]) for b in mb]
                y2 = [float(b["pct_rank_changed"]) for b in mb]
                y3 = [float(b["cumulative_delta_R1"]) for b in mb]
                labels = [str(b["bucket"]) for b in mb]
                fig.suptitle(
                    f"{mrow['method']} | {cell['cell']} | {ds}\n"
                    f"(GT queries: {cell.get('n_queries_with_gt', '')})",
                    fontsize=10,
                )
                _plot_one_method_axes(ax, ax_r, mrow, labels, x, y1, y2, y3)
                meth_tag = str(mrow["method"]).replace(" ", "_").replace("-", "_")
                outp = out_dir / f"NoOp_margin_{ds}_{cell_tag}_{meth_tag}.png"
                fig.savefig(outp, dpi=dpi)
                plt.close(fig)
                print(f"Wrote {outp}", flush=True)
            continue

        fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
        fig.suptitle(
            f"No-op certificate vs margin — {cell['cell']} | {ds}\n"
            f"(queries w/ GT: {cell.get('n_queries_with_gt', '')})",
            fontsize=11,
        )
        for ax, mrow in zip(axes.flat, methods):
            mb = mrow.get("margin_buckets")
            if not mb:
                ax.set_title(f"{mrow['method']} (no margin_buckets in data — re-run script)")
                continue
            x = np.arange(len(mb))
            y1 = [float(b["pct_certified"]) for b in mb]
            y2 = [float(b["pct_rank_changed"]) for b in mb]
            y3 = [float(b["cumulative_delta_R1"]) for b in mb]
            labels = [str(b["bucket"]) for b in mb]
            _plot_one_method_axes(ax, ax.twinx(), mrow, labels, x, y1, y2, y3)

        outp = out_dir / f"NoOp_margin_curves_{ds}_{cell_tag}.png"
        fig.savefig(outp, dpi=dpi)
        plt.close(fig)
        print(f"Wrote {outp}", flush=True)


# ===================================================================== #
# Per-c statistics                                                      #
# ===================================================================== #


def _csls_psi(sims: torch.Tensor, k: int) -> torch.Tensor:
    """psi_eff(c) = r_S(c) / 2, r_S = mean of top-k query->c cosines."""
    k_use = min(k, sims.shape[0])
    r_S = sims.T.topk(k=k_use, dim=1).values.mean(dim=1)
    return r_S / 2.0


def _qbnorm_psi(
    sims: torch.Tensor,
    tau: float,
    query_chunk: int | None = None,
) -> torch.Tensor:
    """
    psi_eff(c) = log Z(c) / tau, Z(c) = sum_q exp(tau * s(q, c)).

    For large N, a single logsumexp(dim=0) can allocate an extra ~N×M workspace and
    OOM when `sims` already fills most of GPU memory. We reduce over query rows in
    chunks and merge with logaddexp (exact).
    """
    tau_f = float(tau)
    n, _m = sims.shape
    qc = int(query_chunk) if query_chunk is not None else 2048
    qc = max(1, min(qc, n))
    if n <= qc:
        return torch.logsumexp(tau_f * sims, dim=0) / tau_f
    acc: torch.Tensor | None = None
    for s in range(0, n, qc):
        e = min(s + qc, n)
        part = torch.logsumexp(tau_f * sims[s:e], dim=0)
        acc = part if acc is None else torch.logaddexp(acc, part)
    assert acc is not None
    return acc / tau_f


def _nnn_psi(sims: torch.Tensor, k: int, w: float) -> torch.Tensor:
    """psi_eff(c) = w * alpha(c), alpha = mean of top-k query->c cosines."""
    k_use = min(k, sims.shape[0])
    alpha = sims.T.topk(k=k_use, dim=1).values.mean(dim=1)
    return float(w) * alpha


# ===================================================================== #
# Method top-1 (full and gated)                                         #
# ===================================================================== #


def _full_top1_offset(sims: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """argmax_c [s(q, c) - psi(c)]. Equivalent argmax for CSLS, QB-Norm, NNN."""
    return (sims - psi.unsqueeze(0)).argmax(dim=1)


def _full_top1_dbnorm(sims: torch.Tensor, psi: torch.Tensor, tau: float) -> torch.Tensor:
    """argmax DB-Norm = argmax [log s + tau*(s - psi)]."""
    eps = 1e-12
    return (
        torch.log(sims.clamp_min(eps)) + float(tau) * (sims - psi.unsqueeze(0))
    ).argmax(dim=1)


def _gated_top1_offset(
    sims: torch.Tensor,
    psi: torch.Tensor,
    cos_top1: torch.Tensor,
    certified: torch.Tensor,
) -> torch.Tensor:
    """Gated top-1: cosine for certified, full method for uncertified."""
    out = cos_top1.clone()
    uncert = (~certified).nonzero(as_tuple=True)[0]
    if uncert.numel() > 0:
        u_top1 = (sims[uncert] - psi.unsqueeze(0)).argmax(dim=1)
        out[uncert] = u_top1
    return out


def _gated_top1_dbnorm(
    sims: torch.Tensor,
    psi: torch.Tensor,
    cos_top1: torch.Tensor,
    certified: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    out = cos_top1.clone()
    uncert = (~certified).nonzero(as_tuple=True)[0]
    if uncert.numel() > 0:
        u_sims = sims[uncert]
        eps = 1e-12
        u_top1 = (
            torch.log(u_sims.clamp_min(eps))
            + float(tau) * (u_sims - psi.unsqueeze(0))
        ).argmax(dim=1)
        out[uncert] = u_top1
    return out


# ===================================================================== #
# Cosine margin and timing helper                                       #
# ===================================================================== #


def _cos_top1_and_margin(sims: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos_top1 [N], m [N] = s(q, c1*) - s(q, c2*))."""
    top2_v, top2_i = sims.topk(2, dim=1)
    return top2_i[:, 0].contiguous(), (top2_v[:, 0] - top2_v[:, 1]).contiguous()


def _correct(top1: np.ndarray, gt_map: dict, qids: list[str]) -> np.ndarray:
    out = np.zeros(len(qids), dtype=bool)
    for i, qid in enumerate(qids):
        gts = set(gt_map.get(qid, []))
        if gts:
            out[i] = int(top1[i]) in gts
    return out


def _time_call(fn, n_warmup: int = 2, n_repeat: int = 5) -> tuple[object, float]:
    """Returns (last_result, mean_seconds_per_call) with CUDA sync."""
    for _ in range(n_warmup):
        _ = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return result, (t1 - t0) / n_repeat


# ===================================================================== #
# Per-method evaluation                                                  #
# ===================================================================== #


def _build_psi(method: str, sims: torch.Tensor, **kw) -> torch.Tensor:
    if method == "CSLS":
        return _csls_psi(sims, k=kw["csls_k"])
    if method == "QB-Norm":
        return _qbnorm_psi(sims, tau=kw["qb_tau"])
    if method == "DB-Norm":
        return _qbnorm_psi(sims, tau=kw["db_tau"])
    if method == "NNN":
        return _nnn_psi(sims, k=kw["nnn_k"], w=kw["nnn_w"])
    raise ValueError(method)


def _full_top1(method: str, sims: torch.Tensor, psi: torch.Tensor, **kw) -> torch.Tensor:
    if method == "DB-Norm":
        return _full_top1_dbnorm(sims, psi, tau=kw["db_tau"])
    return _full_top1_offset(sims, psi)


def _gated_top1(
    method: str,
    sims: torch.Tensor,
    psi: torch.Tensor,
    cos_top1: torch.Tensor,
    certified: torch.Tensor,
    **kw,
) -> torch.Tensor:
    if method == "DB-Norm":
        return _gated_top1_dbnorm(sims, psi, cos_top1, certified, tau=kw["db_tau"])
    return _gated_top1_offset(sims, psi, cos_top1, certified)


def compute_Bq_and_slack(
    psi: torch.Tensor,
    cos_top1: torch.Tensor,
    margin: torch.Tensor,
    cos_topk_idx: torch.Tensor | None,
    cert_local_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Certificate RHS B_q = psi(c1*) - psi_min and slack = margin - B_q (sound global / local-K)."""
    psi_c1 = psi[cos_top1]
    if cert_local_topk >= 2 and cos_topk_idx is not None:
        k_eff = min(int(cert_local_topk), cos_topk_idx.shape[1])
        idx = cos_topk_idx[:, :k_eff]
        psi_min_q = psi[idx].min(dim=1).values
    else:
        psi_min_q = torch.full_like(psi_c1, psi.min())
    B_q = psi_c1 - psi_min_q
    slack = margin - B_q
    return B_q, slack


def _compute_certified_mask(
    psi: torch.Tensor,
    cos_top1: torch.Tensor,
    margin: torch.Tensor,
    cos_topk_idx: torch.Tensor | None,
    cert_local_topk: int,
) -> torch.Tensor:
    """
    Certified if m(q) >= psi(c1*) - psi_min, with psi_min either global min(psi)
    or min over psi on the top-`cert_local_topk` cosine candidates per query.
    """
    B_q, slack = compute_Bq_and_slack(
        psi, cos_top1, margin, cos_topk_idx, cert_local_topk
    )
    return slack >= 0.0


def evaluate_method(
    method: str,
    sims: torch.Tensor,
    cos_top1: torch.Tensor,
    margin: torch.Tensor,
    gt_map: dict,
    qids_list: list[str],
    has_gt: np.ndarray,
    cert_query_export_dir: Path | None = None,
    backbone: str = "",
    direction: str = "",
    cert_local_topk: int = 0,
    cos_topk_idx: torch.Tensor | None = None,
    **kw,
) -> dict:
    # Build psi (used for cert and rerank-only timing).
    psi = _build_psi(method, sims, **kw)

    # Certificate (global min ψ vs candidate-local min over cos top-K).
    certified = _compute_certified_mask(
        psi, cos_top1, margin, cos_topk_idx, cert_local_topk
    )
    n_cert = int(certified.sum().item())
    n_total = int(certified.numel())

    # Compute reference top-1's once, off the timing path.
    full_top1_ref = _full_top1(method, sims, psi, **kw)
    gated_top1_ref = _gated_top1(method, sims, psi, cos_top1, certified, **kw)

    # ---------- end-to-end timing (psi computed inside) ----------
    def fn_full_e2e():
        psi_local = _build_psi(method, sims, **kw)
        return _full_top1(method, sims, psi_local, **kw)

    def fn_gated_e2e():
        psi_local = _build_psi(method, sims, **kw)
        cert_l = _compute_certified_mask(
            psi_local, cos_top1, margin, cos_topk_idx, cert_local_topk
        )
        return _gated_top1(method, sims, psi_local, cos_top1, cert_l, **kw)

    _, full_e2e_s = _time_call(fn_full_e2e)
    _, gated_e2e_s = _time_call(fn_gated_e2e)

    # ---------- rerank-only timing (psi cached) ----------
    def fn_full_rer():
        return _full_top1(method, sims, psi, **kw)

    def fn_gated_rer():
        return _gated_top1(method, sims, psi, cos_top1, certified, **kw)

    _, full_rer_s = _time_call(fn_full_rer)
    _, gated_rer_s = _time_call(fn_gated_rer)

    # ---------- correctness ----------
    cos_top1_np = cos_top1.cpu().numpy()
    full_top1_np = full_top1_ref.cpu().numpy()
    gated_top1_np = gated_top1_ref.cpu().numpy()
    cert_np = certified.cpu().numpy()

    cos_correct = _correct(cos_top1_np, gt_map, qids_list)
    full_correct = _correct(full_top1_np, gt_map, qids_list)
    gated_correct = _correct(gated_top1_np, gt_map, qids_list)

    n_eval = int(has_gt.sum())
    cos_R1 = float(cos_correct[has_gt].mean()) if n_eval else 0.0
    full_R1 = float(full_correct[has_gt].mean()) if n_eval else 0.0
    gated_R1 = float(gated_correct[has_gt].mean()) if n_eval else 0.0

    full_gain = full_R1 - cos_R1
    gated_gain = gated_R1 - cos_R1
    if abs(full_gain) > 1e-9:
        gain_recovery = 100.0 * gated_gain / full_gain
    else:
        gain_recovery = 100.0 if abs(gated_gain) < 1e-9 else 0.0

    n_disagree_total = int((full_top1_np != gated_top1_np).sum())
    n_disagree_certified = int((full_top1_np != gated_top1_np)[cert_np].sum())

    margin_buckets = margin_bucket_rows(
        margin,
        certified,
        cos_top1,
        full_top1_ref,
        cos_correct,
        full_correct,
        has_gt,
    )

    if cert_query_export_dir is not None and str(cert_query_export_dir):
        stem = METHOD_NPZ_STEM.get(method, method.replace("-", "_").replace(" ", "_"))
        fn = Path(cert_query_export_dir) / f"{backbone}__{direction}__{stem}.npz"
        gt_arr = _first_gt_index_array(gt_map, qids_list)
        save_cert_per_query_npz(
            fn,
            margin.detach().cpu().numpy(),
            cert_np,
            cos_top1_np.astype(np.int64),
            full_top1_np.astype(np.int64),
            gt_arr,
        )

    return {
        "method": method,
        "n_queries": n_total,
        "n_certified": n_cert,
        "cert_fraction": n_cert / max(n_total, 1),
        "cos_R1": cos_R1,
        "full_R1": full_R1,
        "gated_R1": gated_R1,
        "full_gain_R1": full_gain,
        "gated_gain_R1": gated_gain,
        "gain_recovery_pct": gain_recovery,
        # soundness sanity: should be 0 if cert is sound (and no exact ties)
        "top1_disagreements_total": n_disagree_total,
        "top1_disagreements_on_certified": n_disagree_certified,
        # timing
        "time_full_e2e_ms": full_e2e_s * 1000,
        "time_gated_e2e_ms": gated_e2e_s * 1000,
        "speedup_e2e": full_e2e_s / max(gated_e2e_s, 1e-12),
        "time_full_rerank_only_ms": full_rer_s * 1000,
        "time_gated_rerank_only_ms": gated_rer_s * 1000,
        "speedup_rerank_only": full_rer_s / max(gated_rer_s, 1e-12),
        "margin_buckets": margin_buckets,
        "cert_local_topk": int(cert_local_topk),
        "cert_mode": "candidate_local" if cert_local_topk >= 2 else "global",
    }


def evaluate_cell(
    dataset: str,
    direction: str,
    backbone: str,
    csls_k: int,
    qb_tau: float,
    db_tau: float,
    nnn_k: int,
    nnn_w: float,
    device: torch.device,
    cert_query_export_dir: Path | None = None,
    cert_local_topk: int = 0,
) -> dict:
    q_emb, g_emb, qids, gids = load_embeddings(dataset, direction, backbone)
    q_emb = F.normalize(q_emb.to(device), dim=1)
    g_emb = F.normalize(g_emb.to(device), dim=1)
    qids_list = [str(x) for x in qids]
    gt_map = build_gt_mapping(qids, gids, direction)
    has_gt = np.array([bool(gt_map.get(q)) for q in qids_list], dtype=bool)

    sims = q_emb @ g_emb.T
    cos_top1, margin = _cos_top1_and_margin(sims)

    cos_topk_idx: torch.Tensor | None = None
    if cert_local_topk >= 2:
        k = min(int(cert_local_topk), sims.shape[1])
        cos_topk_idx = sims.topk(k, dim=1).indices

    cos_correct = _correct(cos_top1.cpu().numpy(), gt_map, qids_list)
    cos_R1 = float(cos_correct[has_gt].mean()) if has_gt.any() else 0.0

    rows = []
    for method in ("CSLS", "QB-Norm", "DB-Norm", "NNN"):
        rows.append(
            evaluate_method(
                method=method,
                sims=sims,
                cos_top1=cos_top1,
                margin=margin,
                gt_map=gt_map,
                qids_list=qids_list,
                has_gt=has_gt,
                cert_query_export_dir=cert_query_export_dir,
                backbone=backbone,
                direction=direction,
                cert_local_topk=cert_local_topk,
                cos_topk_idx=cos_topk_idx,
                csls_k=csls_k,
                qb_tau=qb_tau,
                db_tau=db_tau,
                nnn_k=nnn_k,
                nnn_w=nnn_w,
            )
        )

    return {
        "cell": f"{backbone}|{direction}",
        "n_queries": int(len(qids_list)),
        "n_queries_with_gt": int(has_gt.sum()),
        "n_gallery": int(g_emb.shape[0]),
        "cosine_R1": cos_R1,
        "cert_local_topk": int(cert_local_topk),
        "cert_mode": "candidate_local" if cert_local_topk >= 2 else "global",
        "methods": rows,
    }


# ===================================================================== #
# Markdown output                                                        #
# ===================================================================== #


def write_md(path: Path, results: list[dict], cfg: dict) -> None:
    cert_mode = cfg.get("cert_mode", "global")
    cert_k = int(cfg.get("cert_local_topk", 0))
    lines = [
        "# No-Op Certificate Speedup",
        "",
        f"Dataset: `{cfg['dataset']}` | "
        f"CSLS k: `{cfg['csls_k']}` | QB tau: `{cfg['qb_tau']}` | "
        f"DB tau: `{cfg['db_tau']}` | NNN: `k={cfg['nnn_k']}, w={cfg['nnn_w']}`",
        "",
        f"**Certificate:** `{cert_mode}`"
        + (f" (cos-top-K, K=`{cert_k}`)" if cert_k >= 2 else " (global `min_c psi`)")
        + ".",
        "",
        "- **Global:** `m(q) >= psi_eff(c1*) - min_c psi_eff(c)` over the full gallery (sound).",
        "- **Candidate-local** (`--cert-local-topk K`): `min` only over `cos-top-K(q)`; "
        "surrogate for queries whose refined argmax stays in the shortlist — "
        "expect **%certified in [0,0.01)** to rise vs global; monitor "
        "`top1_disagreements_on_certified`.",
        "",
        "Per-method `psi_eff`:",
        "",
        "- CSLS:    `psi_eff(c) = r_S(c) / 2`",
        "- QB-Norm: `psi_eff(c) = log Z(c) / tau`",
        "- DB-Norm: `psi_eff(c) = log Z(c) / tau` (loose; ignores +log s / tau term)",
        "- NNN:     `psi_eff(c) = w * alpha(c)`",
        "",
        "If certified under the **global** rule, gated top-1 equals full top-1 (sound). "
        "Under **candidate-local** certification, check `top1_disagreements_on_certified`.",
        "",
    ]
    for r in results:
        lines += [
            f"## {r['cell']}",
            "",
            f"- queries: `{r['n_queries']}` (with GT: `{r['n_queries_with_gt']}`) | "
            f"gallery size: `{r['n_gallery']}`",
            f"- cosine R@1: `{r['cosine_R1']:.4f}`",
            "",
            "| Method | %certified | full R@1 | gated R@1 | ΔR@1 vs cos | gain recovery | top1 disagree (cert) | speedup (e2e / rerank) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for m in r["methods"]:
            lines.append(
                f"| {m['method']} | "
                f"{m['cert_fraction']*100:.2f}% | "
                f"{m['full_R1']:.4f} | "
                f"{m['gated_R1']:.4f} | "
                f"{m['gated_gain_R1']:+.4f} | "
                f"{m['gain_recovery_pct']:.2f}% | "
                f"{m['top1_disagreements_total']} ({m['top1_disagreements_on_certified']}) | "
                f"{m['speedup_e2e']:.2f}x / {m['speedup_rerank_only']:.2f}x |"
            )
        lines += [
            "",
            "### Margin buckets (GT queries): % certified | % rank-changed | ΔR@1 in bucket (pp) | cumul. ΔR@1",
            "",
        ]
        for m in r["methods"]:
            lines.append(f"#### {m['method']}")
            lines.append("")
            lines.append("| bucket | n | %cert | %rankΔ | ΔR@1 slice | cumul. ΔR@1 |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for b in m.get("margin_buckets", []):
                lines.append(
                    f"| {b['bucket']} | {b['n']} | {b['pct_certified']:.2f} | "
                    f"{b['pct_rank_changed']:.2f} | {b['delta_R1_in_bucket']:+.5f} | "
                    f"{b['cumulative_delta_R1']:+.5f} |"
                )
            lines.append("")
        lines += [
            "",
            "### Wall-clock (mean over 5 reps after warmup, ms)",
            "",
            "| Method | full e2e | gated e2e | full rerank-only | gated rerank-only |",
            "|---|---:|---:|---:|---:|",
        ]
        for m in r["methods"]:
            lines.append(
                f"| {m['method']} | "
                f"{m['time_full_e2e_ms']:.2f} | "
                f"{m['time_gated_e2e_ms']:.2f} | "
                f"{m['time_full_rerank_only_ms']:.2f} | "
                f"{m['time_gated_rerank_only_ms']:.2f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_margin_curves_json(
    path: Path,
    results: list[dict],
    cfg: dict,
    source_full_json: Path,
) -> None:
    """
    Slim JSON for plotting / sharing: only per-cell metrics + margin_buckets per method.
    Same schema expected by experiments/plot_noop_margin_buckets_from_json.py (config + cells).
    """
    cells_out: list[dict] = []
    for cell in results:
        cells_out.append(
            {
                "cell": cell["cell"],
                "n_queries": cell["n_queries"],
                "n_queries_with_gt": cell["n_queries_with_gt"],
                "n_gallery": cell["n_gallery"],
                "cosine_R1": cell["cosine_R1"],
                "methods": [
                    {
                        "method": m["method"],
                        "n_certified": m["n_certified"],
                        "cert_fraction": m["cert_fraction"],
                        "cos_R1": m["cos_R1"],
                        "full_R1": m["full_R1"],
                        "full_gain_R1": m["full_gain_R1"],
                        "margin_buckets": m["margin_buckets"],
                    }
                    for m in cell["methods"]
                ],
            }
        )
    payload = {
        "description": "No-op certificate margin-bucket series (GT queries). "
        "Fields: pct_certified, pct_rank_changed, delta_R1_in_bucket, cumulative_delta_R1.",
        "source_full_json": str(source_full_json.resolve().as_posix()),
        "generated": cfg.get("generated", datetime.now().isoformat()),
        "config": cfg,
        "margin_bucket_definitions": [
            {
                "low": lo,
                "high": None if hi == float("inf") else hi,
                "label": lab,
            }
            for lo, hi, lab in MARGIN_BUCKET_DEFS
        ],
        "cells": cells_out,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ===================================================================== #
# Main                                                                   #
# ===================================================================== #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="coco_captions")
    ap.add_argument("--backbones", type=str, default="clip,siglip,blip,eva_clip_l14")
    ap.add_argument("--directions", type=str, default="i2t,t2i")
    ap.add_argument("--csls-k", type=int, default=20)
    ap.add_argument("--qb-tau", type=float, default=20.0)
    ap.add_argument("--db-tau", type=float, default=20.0)
    ap.add_argument(
        "--nnn-k",
        type=int,
        default=64,
        help="NNN ψ(c)=w·mean(top-k query→c cosines). Larger k (e.g. 512) smooths α(c); "
        "effective k is min(n_queries, this value).",
    )
    ap.add_argument("--nnn-w", type=float, default=0.5)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument(
        "--margin-plots-dir",
        type=str,
        default="",
        help="If set, save margin-bucket curve PNGs under this directory.",
    )
    ap.add_argument(
        "--margin-plots-one-per-method",
        action="store_true",
        help="With --margin-plots-dir: save one PNG per (cell, method) instead of a 2×2 grid.",
    )
    ap.add_argument(
        "--cert-per-query-dir",
        type=str,
        default="",
        help="If set, write per-query arrays for margin localization: "
        "{backbone}__{direction}__{method}.npz (margin, certified, base_top1, refined_top1, gt).",
    )
    ap.add_argument(
        "--cert-local-topk",
        type=int,
        default=0,
        help="If >=2, use candidate-local cert: m >= psi(c1)-min_{c in cos-top-K} psi(c). "
        "0 = global min_psi (default). Try 20 or 50 for speedup / tight surrogate.",
    )
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    backbones = [x.strip() for x in args.backbones.split(",") if x.strip()]
    directions = [x.strip() for x in args.directions.split(",") if x.strip()]

    cert_npz_dir = Path(args.cert_per_query_dir.strip()) if args.cert_per_query_dir.strip() else None

    results = []
    for b in backbones:
        for d in directions:
            print(f"[cell] {b}|{d}", flush=True)
            results.append(
                evaluate_cell(
                    dataset=args.dataset,
                    direction=d,
                    backbone=b,
                    csls_k=args.csls_k,
                    qb_tau=args.qb_tau,
                    db_tau=args.db_tau,
                    nnn_k=args.nnn_k,
                    nnn_w=args.nnn_w,
                    device=device,
                    cert_query_export_dir=cert_npz_dir,
                    cert_local_topk=args.cert_local_topk,
                )
            )

    cfg = {
        "dataset": args.dataset,
        "backbones": backbones,
        "directions": directions,
        "csls_k": args.csls_k,
        "qb_tau": args.qb_tau,
        "db_tau": args.db_tau,
        "nnn_k": args.nnn_k,
        "nnn_w": args.nnn_w,
        "cert_local_topk": args.cert_local_topk,
        "cert_mode": "candidate_local" if args.cert_local_topk >= 2 else "global",
        "generated": datetime.now().isoformat(),
    }
    out = {"config": cfg, "cells": results}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"NoOpCertificate_{args.dataset}"
    if args.cert_local_topk >= 2:
        stem = f"{stem}_localTopK{args.cert_local_topk}"
    if int(args.nnn_k) != 64:
        stem = f"{stem}_nnnK{int(args.nnn_k)}"
    js = OUTPUT_DIR / f"{stem}.json"
    md = OUTPUT_DIR / f"{stem}.md"
    js.write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_md(md, results, cfg)
    margin_js = OUTPUT_DIR / f"{stem}_margin_curves.json"
    write_margin_curves_json(margin_js, results, cfg, js)
    print(f"Wrote:\n  {js}\n  {md}\n  {margin_js}  (slim; use for plot_noop_margin_buckets_from_json.py)")
    if args.margin_plots_dir.strip():
        plot_margin_bucket_curves(
            results,
            cfg,
            Path(args.margin_plots_dir.strip()),
            one_figure_per_method=args.margin_plots_one_per_method,
        )
    if cert_npz_dir is not None:
        print(f"Per-query cert arrays under: {cert_npz_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
