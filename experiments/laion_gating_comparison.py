#!/usr/bin/env python3
"""
LAION gating-strategy comparison.

For each method M in {CSLS, QB-Norm, DB-Norm, NNN}, compares gating strategies for the rerank step:

  (1) full                baseline — full method top-1 over the entire gallery for every query.
  (2) rerank_cos_topk_pool  always rerank, but argmax is restricted to the cosine top-K pool only
                          (default K=`--rerank-pool-k`, capped by `--topk-cand`).
  (3) selective_top50_pool  **same certificate / skip rule as local_topk** (candidate-local ψ-slack).
                          If reranking: scores restricted to cosine top-K pool only (not full gallery).
                          Use ``--cert-local-strict`` for **m(q) > B_q^local** (slack strictly positive).
  (4) naive_margin        if m(q) >= tau_margin -> cosine top-1; else full method.
                          Heuristic, not sound. tau_margin from --naive-tau (default 0.05).
  (5) exact_global        if m(q) >= psi(c1*) - min_c psi(c) -> cosine; else full.
                          Sound sufficient condition; queries pass deterministically.
  (6) local_topk          if m(q) >= psi(c1*) - min_{c in topK(q)} psi(c) -> cosine; else full.
                          Surrogate: NOT sound (skips outside-topK threats), but in practice
                          usually near-100% gain recovery — typically the most permissive gate.

Per (method, strategy), reports:
  - selected_fraction (% of queries gated to cosine top-1; 0 for full / rerank_cos_topk_pool)
  - gain recovery vs full reranking
        recovery = (R@1_gated - cos_R1) / (R@1_full - cos_R1)
  - top-1 disagreement rate vs full-gallery rerank (`pct_disagree_vs_full_gallery`)
  - wall-clock end-to-end timing (ms), plus speedup vs full-gallery and vs rerank_cos_topk_pool

Optional (--skip-rates): **matched-quantile margin vs ψ-slack** disagreement with full rerank
(same definitions as `experiments/compare_margin_vs_psi_gate.py`). Default targets extend into higher-skip
regimes (e.g. 55–71%) so ψ vs margin can separate where cosine≠full is common.

JSON stores **per-rate aggregates only**, not raw per-query vectors—rerun with different `--skip-rates`
to probe other thresholds without a new experiment conceptually.

Use **full gallery** (`--gallery-splits test,val,train`, `--max-sims-gb` modest) so ψ vs margin
is not degenerate on tiny subsets.

Math (all four methods reduce to argmax_c [s(q,c) - psi_eff(c)] up to monotone
transforms; DB-Norm has an extra +log(s)/tau term that only helps c1*):

  CSLS:    psi_eff(c) = r_S(c) / 2,         r_S(c) = mean_{q in NN_k(c)} s(q,c)
  QB-Norm: psi_eff(c) = log Z(c) / tau,     Z(c)   = sum_q exp(tau * s(q,c))
  DB-Norm: psi_eff(c) = log Z(c) / tau   (loose; sound by monotonicity of +log s)
  NNN:     psi_eff(c) = w * alpha(c)

Soundness of exact_global:
  For c != c1*, gap(c) := s(q,c1*) - s(q,c) >= m(q) (since c2* maximizes s(q,c)
  over c != c1*). So m(q) >= psi(c1*) - psi_min implies gap(c) >= psi(c1*) -
  psi_eff(c) for all c, hence c1* remains argmax.

local_topk drops the "for all c" guarantee: it only takes psi_min over the
cosine top-K candidates, ignoring outside-topK candidates that could in principle
flip top-1. In practice these flips are vanishingly rare because outside-topK
candidates have gap(c) >> m(q).

Default scale: LAION test split (queries=test images, gallery=test texts;
59222 x 59222 ~ 14 GB sims at fp32). For LAION-543K (test queries vs combined
test+val+train texts), pass --gallery-splits test,val,train; this triggers a
batched code path that streams sims and accumulates per-c statistics.

Usage:
  CUDA_VISIBLE_DEVICES=4 python experiments/laion_gating_comparison.py \
      --dataset laion_sample --backbone laion --direction t2i

  # 543K extended gallery (streaming; never materializes N×M sims):
  CUDA_VISIBLE_DEVICES=4 python experiments/laion_gating_comparison.py \
      --dataset laion_sample --backbone laion --direction i2t \
      --gallery-splits test,val,train --batch-size 2048 --max-sims-gb 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[1]

# Reuse the per-c stat / score / argmax helpers from the COCO certificate script.
from psi_slack_pkg.noop_certificate_speedup import (  # noqa: E402
    _csls_psi,
    _qbnorm_psi,
    _nnn_psi,
    _full_top1_offset,
    _full_top1_dbnorm,
    _gated_top1_offset,
    _gated_top1_dbnorm,
    _cos_top1_and_margin,
    _correct,
    _time_call,
)


OUTPUT_DIR = project_root / "evaluation_results" / "tables_GPU"

from psi_slack_pkg.laion import build_pair_gt_mapping, load_laion_embeddings  # noqa: E402


# ===================================================================== #
# Batched compute for the 543K case                                      #
# ===================================================================== #


def compute_sims_full(
    q_emb: torch.Tensor, g_emb: torch.Tensor, batch_size: int
) -> torch.Tensor:
    """sims = q @ g.T, computed in row chunks (peak intermediate ~ batch_size*M*4)."""
    n, m = q_emb.shape[0], g_emb.shape[0]
    sims = torch.empty((n, m), device=q_emb.device, dtype=torch.float32)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        sims[s:e] = q_emb[s:e] @ g_emb.T
    return sims


def cos_top1_margin_batched(sims: torch.Tensor, batch_size: int):
    n = sims.shape[0]
    out_top1 = torch.empty((n,), device=sims.device, dtype=torch.long)
    out_margin = torch.empty((n,), device=sims.device, dtype=sims.dtype)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        v, i = sims[s:e].topk(2, dim=1)
        out_top1[s:e] = i[:, 0]
        out_margin[s:e] = v[:, 0] - v[:, 1]
    return out_top1, out_margin


def cos_topk_batched(sims: torch.Tensor, k: int, batch_size: int) -> torch.Tensor:
    n = sims.shape[0]
    out = torch.empty((n, k), device=sims.device, dtype=torch.long)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        out[s:e] = sims[s:e].topk(k, dim=1).indices
    return out


def _slack_local_tensor(
    psi: torch.Tensor,
    cos_top1: torch.Tensor,
    margin: torch.Tensor,
    cos_topk_idx: torch.Tensor,
    k_local: int,
) -> torch.Tensor:
    """Candidate-local slack — matches noop_certificate_speedup / compare_margin_vs_psi_gate."""
    psi_c1 = psi[cos_top1]
    k_eff = min(k_local, cos_topk_idx.shape[1])
    idx = cos_topk_idx[:, :k_eff]
    psi_min_q = psi[idx].min(dim=1).values
    return margin - (psi_c1 - psi_min_q)


def compute_margin_vs_psi_quantile_curve(
    psi: torch.Tensor,
    cos_top1: torch.Tensor,
    margin: torch.Tensor,
    cos_topk_idx: torch.Tensor,
    full_top1: torch.Tensor,
    cert_local_topk: int,
    skip_rates: list[float],
    *,
    cert_local_strict: bool = False,
) -> list[dict]:
    """
    Matched-skip-rate disagreement vs full rerank (same semantics as compare_margin_vs_psi_gate.py).

    Gates: skip rerank → cosine top-1; else full-method top-1.
    τ_m = quantile_{1−s}(margin), δ_sl = quantile_{1−s}(slack_local).
    """
    slack = _slack_local_tensor(psi, cos_top1, margin, cos_topk_idx, cert_local_topk)
    margin_np = margin.detach().cpu().numpy().astype(np.float64)
    slack_np = slack.detach().cpu().numpy().astype(np.float64)
    cos_np = cos_top1.cpu().numpy().astype(np.int64)
    full_np = full_top1.detach().cpu().numpy().astype(np.int64)
    rows: list[dict] = []
    for s in skip_rates:
        if not (0.0 < s < 1.0):
            continue
        tau = float(np.quantile(margin_np, 1.0 - s))
        delta = float(np.quantile(slack_np, 1.0 - s))
        skip_m = margin_np >= tau
        if cert_local_strict:
            skip_psi = slack_np > delta
        else:
            skip_psi = slack_np >= delta
        gated_m = np.where(skip_m, cos_np, full_np)
        gated_psi = np.where(skip_psi, cos_np, full_np)
        disagree_m = float(np.mean(gated_m != full_np))
        disagree_psi = float(np.mean(gated_psi != full_np))
        rows.append(
            {
                "skip_rate_target": float(s),
                "tau_margin": tau,
                "delta_slack_local": delta,
                "skip_rate_empirical_margin": float(np.mean(skip_m)),
                "skip_rate_empirical_psi": float(np.mean(skip_psi)),
                "pct_disagree_with_full_margin_gate": disagree_m,
                "pct_disagree_with_full_psi_gate": disagree_psi,
                "delta_disagree_psi_minus_margin_pp": float(
                    100.0 * (disagree_m - disagree_psi)
                ),
                "n_queries": int(margin_np.shape[0]),
            }
        )
    return rows


# ===================================================================== #
# Strategies                                                              #
# ===================================================================== #


METHODS = ("CSLS", "QB-Norm", "DB-Norm", "NNN")
STRATEGIES = (
    "full",
    "rerank_cos_topk_pool",
    "selective_top50_pool",
    "selective_top50_pool_exact_global",
    "selective_top50_pool_maxform",
    "naive_margin",
    "exact_global",
    "local_topk",
)


def _rhs_Bq_local(psi: torch.Tensor, cos_top1: torch.Tensor, cos_topk_idx: torch.Tensor) -> torch.Tensor:
    """Certificate RHS: psi(c1*) - min_{c in cos-top-K} psi(c)."""
    psi_topk = psi[cos_topk_idx]
    psi_min_topk = psi_topk.min(dim=1).values
    return psi[cos_top1] - psi_min_topk


def _local_topk_margin_certified(
    margin: torch.Tensor,
    psi: torch.Tensor,
    cos_top1: torch.Tensor,
    cos_topk_idx: torch.Tensor,
    *,
    strict: bool,
) -> torch.Tensor:
    """Certified iff m(q) > B_q^local (strict) or m(q) >= B_q^local (default)."""
    rhs = _rhs_Bq_local(psi, cos_top1, cos_topk_idx)
    return margin > rhs if strict else margin >= rhs


def _global_margin_certified(
    margin: torch.Tensor,
    psi: torch.Tensor,
    cos_top1: torch.Tensor,
    *,
    strict: bool,
) -> torch.Tensor:
    """Gallery-global ψ slack: m(q) > ψ(c1*) − min_gallery ψ (strict) or >= (default)."""
    rhs = psi[cos_top1] - psi.min()
    return margin > rhs if strict else margin >= rhs


def _local_topk_maxform_certified(
    cos_topk_sims: torch.Tensor,
    psi_pack: torch.Tensor,
    *,
    strict: bool,
) -> torch.Tensor:
    """
    Prop. 2 candidate-local max-form on cosine shortlist C_K(q):

        max_{c ∈ C_K \\ {c1}} [ ψ(c1*) − ψ(c) − ( s(q,c1*) − s(q,c) ) ] < 0

    `cos_topk_sims[q,k] = s(q, c_k)` and `psi_pack[q,k] = ψ(c_k)` with column 0 = c1*.
    """
    gap_cos = cos_topk_sims[:, :1] - cos_topk_sims
    gap_psi = psi_pack[:, :1] - psi_pack
    term = gap_psi - gap_cos
    n, k = term.shape
    if k <= 1:
        return torch.ones(n, dtype=torch.bool, device=term.device)
    neg_inf = torch.tensor(float("-inf"), device=term.device, dtype=term.dtype)
    term = term.clone()
    term[:, 0] = neg_inf
    max_term = term[:, 1:].max(dim=1).values
    return max_term < 0 if strict else max_term <= 0


def _dense_cos_topk_sims(sims: torch.Tensor, cos_topk_idx: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(sims.shape[0], device=sims.device, dtype=torch.long).unsqueeze(1).expand_as(
        cos_topk_idx
    )
    return sims[rows, cos_topk_idx]


def _build_psi(method: str, sims: torch.Tensor, **kw) -> torch.Tensor:
    qchunk = kw.get("qb_psi_query_chunk")
    if method == "CSLS":
        return _csls_psi(sims, k=kw["csls_k"])
    if method == "QB-Norm":
        return _qbnorm_psi(sims, tau=kw["qb_tau"], query_chunk=qchunk)
    if method == "DB-Norm":
        return _qbnorm_psi(sims, tau=kw["db_tau"], query_chunk=qchunk)
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


def _certified_mask(
    strategy: str,
    psi: torch.Tensor,
    cos_top1: torch.Tensor,
    margin: torch.Tensor,
    cos_topk_idx: torch.Tensor,
    naive_tau: float,
    *,
    cert_local_strict: bool = False,
    cos_topk_sims: torch.Tensor | None = None,
) -> torch.Tensor:
    if strategy == "full":
        return torch.zeros_like(margin, dtype=torch.bool)
    if strategy == "naive_margin":
        return margin >= float(naive_tau)
    if strategy == "exact_global":
        return margin >= (psi[cos_top1] - psi.min())
    if strategy == "local_topk":
        return _local_topk_margin_certified(
            margin, psi, cos_top1, cos_topk_idx, strict=cert_local_strict
        )
    if strategy == "selective_top50_pool_exact_global":
        return _global_margin_certified(
            margin, psi, cos_top1, strict=cert_local_strict
        )
    if strategy == "selective_top50_pool_maxform":
        if cos_topk_sims is None:
            raise ValueError("selective_top50_pool_maxform requires cos_topk_sims")
        psi_pack = psi[cos_topk_idx]
        return _local_topk_maxform_certified(
            cos_topk_sims, psi_pack, strict=cert_local_strict
        )
    raise ValueError(strategy)


def _cos_top_pool_top1(
    method: str,
    sims: torch.Tensor,
    psi: torch.Tensor,
    cos_topk_idx: torch.Tensor,
    pool_k: int,
    **kw,
) -> torch.Tensor:
    """Argmax of method scores restricted to cosine top-`pool_k` gallery indices."""
    pk = min(int(pool_k), cos_topk_idx.shape[1])
    idx = cos_topk_idx[:, :pk]
    sim = torch.gather(sims, 1, idx)
    pc = psi[idx]
    if method == "DB-Norm":
        eps = 1e-12
        tau = float(kw["db_tau"])
        scores = torch.log(sim.clamp_min(eps)) + tau * (sim - pc)
    else:
        scores = sim - pc
    inner = scores.argmax(dim=1)
    return idx[torch.arange(sims.shape[0], device=sims.device), inner]


def _gated_top1_pool(
    method: str,
    sims: torch.Tensor,
    psi: torch.Tensor,
    cos_top1: torch.Tensor,
    certified: torch.Tensor,
    cos_topk_idx: torch.Tensor,
    pool_k: int,
    **kw,
) -> torch.Tensor:
    """Certified queries → cosine top-1; uncertified → method top-1 over cosine top-`pool_k` only."""
    out = cos_top1.clone()
    uncert = (~certified).nonzero(as_tuple=True)[0]
    if uncert.numel() == 0:
        return out
    sub = _cos_top_pool_top1(
        method,
        sims[uncert],
        psi,
        cos_topk_idx[uncert],
        pool_k,
        **kw,
    )
    out[uncert] = sub
    return out


def evaluate_strategy(
    method: str,
    strategy: str,
    sims: torch.Tensor,
    psi: torch.Tensor,
    cos_top1: torch.Tensor,
    margin: torch.Tensor,
    cos_topk_idx: torch.Tensor,
    naive_tau: float,
    rerank_pool_k: int,
    *,
    cert_local_strict: bool = False,
    **method_kw,
) -> dict:
    pool_k = min(int(rerank_pool_k), int(cos_topk_idx.shape[1]))
    # Reference top-1 (off the timing path).
    if strategy == "full":
        ref_top1 = _full_top1(method, sims, psi, **method_kw)
        cert_mask = torch.zeros_like(margin, dtype=torch.bool)
    elif strategy == "rerank_cos_topk_pool":
        ref_top1 = _cos_top_pool_top1(method, sims, psi, cos_topk_idx, pool_k, **method_kw)
        cert_mask = torch.zeros_like(margin, dtype=torch.bool)
    elif strategy == "selective_top50_pool":
        cert_mask = _certified_mask(
            "local_topk",
            psi,
            cos_top1,
            margin,
            cos_topk_idx,
            naive_tau,
            cert_local_strict=cert_local_strict,
        )
        ref_top1 = _gated_top1_pool(
            method,
            sims,
            psi,
            cos_top1,
            cert_mask,
            cos_topk_idx,
            pool_k,
            **method_kw,
        )
    elif strategy == "selective_top50_pool_exact_global":
        cert_mask = _certified_mask(
            "selective_top50_pool_exact_global",
            psi,
            cos_top1,
            margin,
            cos_topk_idx,
            naive_tau,
            cert_local_strict=cert_local_strict,
        )
        ref_top1 = _gated_top1_pool(
            method,
            sims,
            psi,
            cos_top1,
            cert_mask,
            cos_topk_idx,
            pool_k,
            **method_kw,
        )
    elif strategy == "selective_top50_pool_maxform":
        cos_topk_sims = _dense_cos_topk_sims(sims, cos_topk_idx)
        cert_mask = _certified_mask(
            "selective_top50_pool_maxform",
            psi,
            cos_top1,
            margin,
            cos_topk_idx,
            naive_tau,
            cert_local_strict=cert_local_strict,
            cos_topk_sims=cos_topk_sims,
        )
        ref_top1 = _gated_top1_pool(
            method,
            sims,
            psi,
            cos_top1,
            cert_mask,
            cos_topk_idx,
            pool_k,
            **method_kw,
        )
    else:
        cert_mask = _certified_mask(
            strategy,
            psi,
            cos_top1,
            margin,
            cos_topk_idx,
            naive_tau,
            cert_local_strict=cert_local_strict,
        )
        ref_top1 = _gated_top1(method, sims, psi, cos_top1, cert_mask, **method_kw)

    # Wall-clock end-to-end (psi cached; cert + score on uncertified).
    if strategy == "full":
        def fn():
            return _full_top1(method, sims, psi, **method_kw)

    elif strategy == "rerank_cos_topk_pool":
        def fn():
            return _cos_top_pool_top1(method, sims, psi, cos_topk_idx, pool_k, **method_kw)

    elif strategy == "selective_top50_pool":
        def fn():
            mask = _local_topk_margin_certified(
                margin, psi, cos_top1, cos_topk_idx, strict=cert_local_strict
            )
            return _gated_top1_pool(
                method,
                sims,
                psi,
                cos_top1,
                mask,
                cos_topk_idx,
                pool_k,
                **method_kw,
            )

    elif strategy == "selective_top50_pool_exact_global":
        def fn():
            mask = _global_margin_certified(
                margin, psi, cos_top1, strict=cert_local_strict
            )
            return _gated_top1_pool(
                method,
                sims,
                psi,
                cos_top1,
                mask,
                cos_topk_idx,
                pool_k,
                **method_kw,
            )

    elif strategy == "selective_top50_pool_maxform":
        cos_topk_sims_tf = _dense_cos_topk_sims(sims, cos_topk_idx)
        psi_pack_tf = psi[cos_topk_idx]

        def fn():
            mask = _local_topk_maxform_certified(
                cos_topk_sims_tf, psi_pack_tf, strict=cert_local_strict
            )
            return _gated_top1_pool(
                method,
                sims,
                psi,
                cos_top1,
                mask,
                cos_topk_idx,
                pool_k,
                **method_kw,
            )

    elif strategy == "naive_margin":
        def fn():
            mask = margin >= float(naive_tau)
            return _gated_top1(method, sims, psi, cos_top1, mask, **method_kw)
    elif strategy == "exact_global":
        def fn():
            mask = margin >= (psi[cos_top1] - psi.min())
            return _gated_top1(method, sims, psi, cos_top1, mask, **method_kw)
    elif strategy == "local_topk":
        def fn():
            mask = _local_topk_margin_certified(
                margin, psi, cos_top1, cos_topk_idx, strict=cert_local_strict
            )
            return _gated_top1(method, sims, psi, cos_top1, mask, **method_kw)
    else:
        raise ValueError(strategy)

    _, t_e2e_s = _time_call(fn)

    return {
        "method": method,
        "strategy": strategy,
        "selected_fraction": float(cert_mask.float().mean().item()),
        "n_selected": int(cert_mask.sum().item()),
        "n_total": int(cert_mask.numel()),
        "top1_ref": ref_top1,
        "time_e2e_ms": t_e2e_s * 1000,
    }


# ===================================================================== #
# Streaming evaluation (gallery too large for N×M sims tensor)           #
# ===================================================================== #


def _streaming_psi_vectors(
    q_emb: torch.Tensor,
    g_emb: torch.Tensor,
    n_q: int,
    n_g: int,
    csls_k: int,
    qb_tau: float,
    nnn_k: int,
    nnn_w: float,
    gallery_chunk: int,
    qb_psi_query_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One pass over gallery chunks; returns psi for CSLS, QB/DB, NNN."""
    device = q_emb.device
    psi_csls = torch.empty(n_g, device=device, dtype=torch.float32)
    psi_qb = torch.empty(n_g, device=device, dtype=torch.float32)
    psi_nnn = torch.empty(n_g, device=device, dtype=torch.float32)
    qc = max(1, min(int(qb_psi_query_chunk), n_q))
    g_ch = max(1, int(gallery_chunk))

    for j0 in range(0, n_g, g_ch):
        j1 = min(j0 + g_ch, n_g)
        sim = q_emb @ g_emb[j0:j1].T
        kc = min(int(csls_k), n_q)
        psi_csls[j0:j1] = sim.T.topk(kc, dim=1).values.mean(dim=1) / 2.0
        kn = min(int(nnn_k), n_q)
        psi_nnn[j0:j1] = float(nnn_w) * sim.T.topk(kn, dim=1).values.mean(dim=1)
        acc: torch.Tensor | None = None
        for s in range(0, n_q, qc):
            e = min(s + qc, n_q)
            part = torch.logsumexp(float(qb_tau) * sim[s:e], dim=0)
            acc = part if acc is None else torch.logaddexp(acc, part)
        assert acc is not None
        psi_qb[j0:j1] = acc / float(qb_tau)

    return psi_csls, psi_qb, psi_nnn


def _streaming_cosine_top1_margin_topk(
    q_emb: torch.Tensor,
    g_emb: torch.Tensor,
    n_q: int,
    n_g: int,
    topk_cand: int,
    query_chunk: int,
    gallery_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = q_emb.device
    dtype = q_emb.dtype
    cos_top1 = torch.empty(n_q, device=device, dtype=torch.long)
    margin = torch.empty(n_q, device=device, dtype=dtype)
    cos_topk_idx = torch.empty(n_q, topk_cand, device=device, dtype=torch.long)
    g_ch = max(1, int(gallery_chunk))
    q_ch = max(1, int(query_chunk))
    K = int(topk_cand)

    for s0 in range(0, n_q, q_ch):
        s1 = min(s0 + q_ch, n_q)
        qb = q_emb[s0:s1]
        bq = s1 - s0
        g1 = torch.full((bq,), -float("inf"), device=device, dtype=dtype)
        g2 = torch.full((bq,), -float("inf"), device=device, dtype=dtype)
        i1 = torch.zeros(bq, device=device, dtype=torch.long)
        i2 = torch.zeros(bq, device=device, dtype=torch.long)
        topk_vals: torch.Tensor | None = None
        topk_idx: torch.Tensor | None = None

        for j0 in range(0, n_g, g_ch):
            j1 = min(j0 + g_ch, n_g)
            sim = qb @ g_emb[j0:j1].T
            bc = sim.shape[1]
            gj = torch.arange(j0, j1, device=device, dtype=torch.long).unsqueeze(0).expand(bq, -1)

            nk = min(2, bc)
            cv, cix = sim.topk(nk, dim=1)
            c1 = cv[:, 0]
            if nk >= 2:
                c2 = cv[:, 1]
                l1 = cix[:, 0]
                l2 = cix[:, 1]
            else:
                c2 = torch.full((bq,), -float("inf"), device=device, dtype=dtype)
                l1 = cix[:, 0]
                l2 = torch.zeros(bq, device=device, dtype=torch.long)

            mval = torch.stack([g1, g2, c1, c2], dim=1)
            midx = torch.stack([i1, i2, j0 + l1, j0 + l2], dim=1)
            tv, tk = mval.topk(2, dim=1)
            g1, g2 = tv[:, 0], tv[:, 1]
            i1 = torch.gather(midx, 1, tk[:, 0:1]).squeeze(1)
            i2 = torch.gather(midx, 1, tk[:, 1:2]).squeeze(1)

            take = min(K, bc)
            lv, lix = sim.topk(take, dim=1)
            lglob = torch.gather(gj, 1, lix)
            if topk_vals is None:
                topk_vals = lv
                topk_idx = lglob
            else:
                merged_v = torch.cat([topk_vals, lv], dim=1)
                merged_i = torch.cat([topk_idx, lglob], dim=1)
                k_use = min(K, merged_v.shape[1])
                tv2, tk2 = merged_v.topk(k_use, dim=1)
                topk_vals = tv2
                topk_idx = torch.gather(merged_i, 1, tk2)

        cos_top1[s0:s1] = i1
        margin[s0:s1] = g1 - g2
        if topk_idx is not None:
            filled = topk_idx.shape[1]
            cos_topk_idx[s0:s1, :filled] = topk_idx
            if filled < K:
                cos_topk_idx[s0:s1, filled:K] = topk_idx[:, -1:].expand(-1, K - filled)
        else:
            cos_topk_idx[s0:s1] = 0

    return cos_top1, margin, cos_topk_idx


def _streaming_gather_cos_topk_sim(
    q_emb: torch.Tensor,
    g_emb: torch.Tensor,
    cos_topk_idx: torch.Tensor,
    query_chunk: int,
) -> torch.Tensor:
    """Cosine similarities s(q,c) for each shortlist slot; shape [n_q, K]."""
    n_q, K = cos_topk_idx.shape
    device = q_emb.device
    dtype = q_emb.dtype
    out = torch.empty((n_q, K), device=device, dtype=dtype)
    q_ch = max(1, int(query_chunk))

    for s0 in range(0, n_q, q_ch):
        s1 = min(s0 + q_ch, n_q)
        qb = q_emb[s0:s1]
        bq = s1 - s0
        idx_block = cos_topk_idx[s0:s1]
        g_sel = g_emb[idx_block.reshape(-1)].reshape(bq, K, -1)
        out[s0:s1] = (qb.unsqueeze(1) * g_sel).sum(dim=-1)
    return out


def _streaming_full_top1(
    method: str,
    q_emb: torch.Tensor,
    g_emb: torch.Tensor,
    n_q: int,
    n_g: int,
    psi: torch.Tensor,
    query_chunk: int,
    gallery_chunk: int,
    db_tau: float,
) -> torch.Tensor:
    device = q_emb.device
    dtype = q_emb.dtype
    out = torch.empty(n_q, device=device, dtype=torch.long)
    g_ch = max(1, int(gallery_chunk))
    q_ch = max(1, int(query_chunk))
    eps = 1e-12

    for s0 in range(0, n_q, q_ch):
        s1 = min(s0 + q_ch, n_q)
        qb = q_emb[s0:s1]
        bq = s1 - s0
        best = torch.full((bq,), -float("inf"), device=device, dtype=dtype)
        best_j = torch.zeros(bq, device=device, dtype=torch.long)
        for j0 in range(0, n_g, g_ch):
            j1 = min(j0 + g_ch, n_g)
            sim = qb @ g_emb[j0:j1].T
            pc = psi[j0:j1]
            if method == "DB-Norm":
                scores = torch.log(sim.clamp_min(eps)) + float(db_tau) * (sim - pc.unsqueeze(0))
            else:
                scores = sim - pc.unsqueeze(0)
            mv, mi = scores.max(dim=1)
            bet = mv > best
            best = torch.where(bet, mv, best)
            best_j = torch.where(bet, j0 + mi, best_j)
        out[s0:s1] = best_j
    return out


def _streaming_cos_top_pool_top1(
    method: str,
    q_emb: torch.Tensor,
    g_emb: torch.Tensor,
    cos_topk_idx: torch.Tensor,
    pool_k: int,
    psi: torch.Tensor,
    n_q: int,
    query_chunk: int,
    db_tau: float,
) -> torch.Tensor:
    """Argmax of method scores on cosine top-`pool_k` indices only (streaming)."""
    pk = min(int(pool_k), int(cos_topk_idx.shape[1]))
    device = q_emb.device
    dtype = q_emb.dtype
    out = torch.empty(n_q, device=device, dtype=torch.long)
    q_ch = max(1, int(query_chunk))
    eps = 1e-12

    for s0 in range(0, n_q, q_ch):
        s1 = min(s0 + q_ch, n_q)
        qb = q_emb[s0:s1]
        bq = s1 - s0
        idx_block = cos_topk_idx[s0:s1, :pk]
        g_sel = g_emb[idx_block.reshape(-1)].reshape(bq, pk, -1)
        sim = (qb.unsqueeze(1) * g_sel).sum(dim=-1)
        pc = psi[idx_block]
        if method == "DB-Norm":
            scores = torch.log(sim.clamp_min(eps)) + float(db_tau) * (sim - pc)
        else:
            scores = sim - pc
        inner = scores.argmax(dim=1)
        rows = torch.arange(bq, device=device)
        out[s0:s1] = idx_block[rows, inner]
    return out


def _streaming_gated_top1(
    method: str,
    q_emb: torch.Tensor,
    g_emb: torch.Tensor,
    n_g: int,
    psi: torch.Tensor,
    cos_top1: torch.Tensor,
    certified: torch.Tensor,
    query_chunk: int,
    gallery_chunk: int,
    db_tau: float,
) -> torch.Tensor:
    out = cos_top1.clone()
    uncert = (~certified).nonzero(as_tuple=True)[0]
    if uncert.numel() == 0:
        return out
    device = q_emb.device
    dtype = q_emb.dtype
    g_ch = max(1, int(gallery_chunk))
    q_ch = max(1, int(query_chunk))
    eps = 1e-12

    for a in range(0, uncert.numel(), q_ch):
        sub = uncert[a : a + q_ch]
        qb = q_emb[sub]
        bq = sub.shape[0]
        best = torch.full((bq,), -float("inf"), device=device, dtype=dtype)
        best_j = torch.zeros(bq, device=device, dtype=torch.long)
        for j0 in range(0, n_g, g_ch):
            j1 = min(j0 + g_ch, n_g)
            sim = qb @ g_emb[j0:j1].T
            pc = psi[j0:j1]
            if method == "DB-Norm":
                scores = torch.log(sim.clamp_min(eps)) + float(db_tau) * (sim - pc.unsqueeze(0))
            else:
                scores = sim - pc.unsqueeze(0)
            mv, mi = scores.max(dim=1)
            bet = mv > best
            best = torch.where(bet, mv, best)
            best_j = torch.where(bet, j0 + mi, best_j)
        out[sub] = best_j
    return out


def _streaming_gated_top1_pool(
    method: str,
    q_emb: torch.Tensor,
    g_emb: torch.Tensor,
    psi: torch.Tensor,
    cos_top1: torch.Tensor,
    certified: torch.Tensor,
    cos_topk_idx: torch.Tensor,
    pool_k: int,
    query_chunk: int,
    db_tau: float,
) -> torch.Tensor:
    """local_topk certificate; uncertified queries reranked on cosine top-`pool_k` only."""
    out = cos_top1.clone()
    uncert = (~certified).nonzero(as_tuple=True)[0]
    if uncert.numel() == 0:
        return out
    q_sub = q_emb[uncert]
    idx_sub = cos_topk_idx[uncert]
    nu = q_sub.shape[0]
    sub = _streaming_cos_top_pool_top1(
        method,
        q_sub,
        g_emb,
        idx_sub,
        pool_k,
        psi,
        nu,
        query_chunk,
        db_tau,
    )
    out[uncert] = sub
    return out


def evaluate_cell_streaming(
    q_emb: torch.Tensor,
    g_emb: torch.Tensor,
    q_ids: np.ndarray,
    g_ids: np.ndarray,
    backbone: str,
    direction: str,
    gallery_splits: list[str],
    csls_k: int,
    qb_tau: float,
    db_tau: float,
    nnn_k: int,
    nnn_w: float,
    naive_tau: float,
    topk_cand: int,
    gallery_chunk: int,
    query_chunk: int,
    qb_psi_query_chunk: int,
    skip_rates: list[float],
    rerank_pool_k: int,
    device: torch.device,
    *,
    cert_local_strict: bool = False,
) -> dict:
    print("  [streaming] gallery × queries too large for dense sims; multi-pass chunked matmul", flush=True)
    n_q, n_g = q_emb.shape[0], g_emb.shape[0]
    print(f"  queries={n_q}  gallery={n_g}", flush=True)

    qids_list = [str(x) for x in q_ids]
    gt_map = build_pair_gt_mapping(q_ids, g_ids, direction)
    has_gt = np.array([bool(gt_map.get(q)) for q in qids_list], dtype=bool)
    print(f"  queries_with_gt={int(has_gt.sum())}", flush=True)

    print("  streaming: pass 1 — per-c psi (CSLS, QB/DB, NNN) ...", flush=True)
    t0 = time.perf_counter()
    psi_csls, psi_qb, psi_nnn = _streaming_psi_vectors(
        q_emb,
        g_emb,
        n_q,
        n_g,
        csls_k,
        qb_tau,
        nnn_k,
        nnn_w,
        gallery_chunk,
        qb_psi_query_chunk,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"    psi done in {time.perf_counter() - t0:.2f}s", flush=True)

    print("  streaming: pass 2 — cosine top-1, margin, local top-K ...", flush=True)
    t1 = time.perf_counter()
    cos_top1, margin, cos_topk_idx = _streaming_cosine_top1_margin_topk(
        q_emb, g_emb, n_q, n_g, topk_cand, query_chunk, gallery_chunk
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"    cosine stats in {time.perf_counter() - t1:.2f}s", flush=True)

    print("  streaming: pass 2b — cosine sims on cosine shortlist ...", flush=True)
    t1b = time.perf_counter()
    cos_topk_sims = _streaming_gather_cos_topk_sim(
        q_emb, g_emb, cos_topk_idx, query_chunk
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"    shortlist sims in {time.perf_counter() - t1b:.2f}s", flush=True)

    cos_top1_np = cos_top1.cpu().numpy()
    cos_correct = _correct(cos_top1_np, gt_map, qids_list)
    cos_R1 = float(cos_correct[has_gt].mean()) if has_gt.any() else 0.0
    print(f"  cos R@1 = {cos_R1:.4f}", flush=True)

    pool_k = min(int(rerank_pool_k), int(cos_topk_idx.shape[1]))

    method_psi = {
        "CSLS": psi_csls,
        "QB-Norm": psi_qb,
        "DB-Norm": psi_qb,
        "NNN": psi_nnn,
    }

    rows: list[dict] = []
    quantile_skip_margin_vs_psi: dict[str, list[dict]] = {}

    for method in METHODS:
        print(f"  method = {method} (streaming top-1)", flush=True)
        psi = method_psi[method]
        t_full = time.perf_counter()
        full_top1 = _streaming_full_top1(
            method, q_emb, g_emb, n_q, n_g, psi, query_chunk, gallery_chunk, db_tau
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"    full top-1 pass in {time.perf_counter() - t_full:.2f}s", flush=True)
        full_top1_np = full_top1.cpu().numpy()
        full_correct = _correct(full_top1_np, gt_map, qids_list)
        full_R1 = float(full_correct[has_gt].mean()) if has_gt.any() else 0.0
        full_gain = full_R1 - cos_R1

        if skip_rates:
            k_cert = min(topk_cand, int(cos_topk_idx.shape[1]))
            quantile_skip_margin_vs_psi[method] = compute_margin_vs_psi_quantile_curve(
                psi,
                cos_top1,
                margin,
                cos_topk_idx,
                full_top1,
                k_cert,
                skip_rates,
                cert_local_strict=cert_local_strict,
            )

        cert_scalar_mask = _local_topk_margin_certified(
            margin,
            psi,
            cos_top1,
            cos_topk_idx,
            strict=cert_local_strict,
        )
        scalar_sel_top1 = _streaming_gated_top1_pool(
            method,
            q_emb,
            g_emb,
            psi,
            cos_top1,
            cert_scalar_mask,
            cos_topk_idx,
            pool_k,
            query_chunk,
            db_tau,
        )
        psi_pack_m = psi[cos_topk_idx]
        cert_maxform_mask = _local_topk_maxform_certified(
            cos_topk_sims, psi_pack_m, strict=cert_local_strict
        )
        maxform_sel_top1 = _streaming_gated_top1_pool(
            method,
            q_emb,
            g_emb,
            psi,
            cos_top1,
            cert_maxform_mask,
            cos_topk_idx,
            pool_k,
            query_chunk,
            db_tau,
        )
        pct_dis_maxform_vs_scalar = float(
            100.0
            * float((maxform_sel_top1 != scalar_sel_top1).float().mean().item())
        )

        for strategy in STRATEGIES:
            if strategy == "full":
                cert_mask = torch.zeros_like(margin, dtype=torch.bool)
                ref_top1 = full_top1
            elif strategy == "rerank_cos_topk_pool":
                cert_mask = torch.zeros_like(margin, dtype=torch.bool)
                ref_top1 = _streaming_cos_top_pool_top1(
                    method,
                    q_emb,
                    g_emb,
                    cos_topk_idx,
                    pool_k,
                    psi,
                    n_q,
                    query_chunk,
                    db_tau,
                )
            elif strategy == "selective_top50_pool":
                cert_mask = _certified_mask(
                    "local_topk",
                    psi,
                    cos_top1,
                    margin,
                    cos_topk_idx,
                    naive_tau,
                    cert_local_strict=cert_local_strict,
                )
                ref_top1 = _streaming_gated_top1_pool(
                    method,
                    q_emb,
                    g_emb,
                    psi,
                    cos_top1,
                    cert_mask,
                    cos_topk_idx,
                    pool_k,
                    query_chunk,
                    db_tau,
                )
            elif strategy == "selective_top50_pool_exact_global":
                cert_mask = _certified_mask(
                    "selective_top50_pool_exact_global",
                    psi,
                    cos_top1,
                    margin,
                    cos_topk_idx,
                    naive_tau,
                    cert_local_strict=cert_local_strict,
                )
                ref_top1 = _streaming_gated_top1_pool(
                    method,
                    q_emb,
                    g_emb,
                    psi,
                    cos_top1,
                    cert_mask,
                    cos_topk_idx,
                    pool_k,
                    query_chunk,
                    db_tau,
                )
            elif strategy == "selective_top50_pool_maxform":
                cert_mask = _certified_mask(
                    "selective_top50_pool_maxform",
                    psi,
                    cos_top1,
                    margin,
                    cos_topk_idx,
                    naive_tau,
                    cert_local_strict=cert_local_strict,
                    cos_topk_sims=cos_topk_sims,
                )
                ref_top1 = _streaming_gated_top1_pool(
                    method,
                    q_emb,
                    g_emb,
                    psi,
                    cos_top1,
                    cert_mask,
                    cos_topk_idx,
                    pool_k,
                    query_chunk,
                    db_tau,
                )
            else:
                cert_mask = _certified_mask(
                    strategy,
                    psi,
                    cos_top1,
                    margin,
                    cos_topk_idx,
                    naive_tau,
                    cert_local_strict=cert_local_strict,
                )
                ref_top1 = _streaming_gated_top1(
                    method,
                    q_emb,
                    g_emb,
                    n_g,
                    psi,
                    cos_top1,
                    cert_mask,
                    query_chunk,
                    gallery_chunk,
                    db_tau,
                )

            if strategy == "full":

                def fn():
                    return _streaming_full_top1(
                        method, q_emb, g_emb, n_q, n_g, psi, query_chunk, gallery_chunk, db_tau
                    )

            elif strategy == "rerank_cos_topk_pool":

                def fn():
                    return _streaming_cos_top_pool_top1(
                        method,
                        q_emb,
                        g_emb,
                        cos_topk_idx,
                        pool_k,
                        psi,
                        n_q,
                        query_chunk,
                        db_tau,
                    )

            elif strategy == "selective_top50_pool":

                def fn():
                    mask = _local_topk_margin_certified(
                        margin,
                        psi,
                        cos_top1,
                        cos_topk_idx,
                        strict=cert_local_strict,
                    )
                    return _streaming_gated_top1_pool(
                        method,
                        q_emb,
                        g_emb,
                        psi,
                        cos_top1,
                        mask,
                        cos_topk_idx,
                        pool_k,
                        query_chunk,
                        db_tau,
                    )

            elif strategy == "selective_top50_pool_maxform":

                def fn():
                    mask = _local_topk_maxform_certified(
                        cos_topk_sims,
                        psi[cos_topk_idx],
                        strict=cert_local_strict,
                    )
                    return _streaming_gated_top1_pool(
                        method,
                        q_emb,
                        g_emb,
                        psi,
                        cos_top1,
                        mask,
                        cos_topk_idx,
                        pool_k,
                        query_chunk,
                        db_tau,
                    )

            elif strategy == "selective_top50_pool_exact_global":

                def fn():
                    mask = _global_margin_certified(
                        margin,
                        psi,
                        cos_top1,
                        strict=cert_local_strict,
                    )
                    return _streaming_gated_top1_pool(
                        method,
                        q_emb,
                        g_emb,
                        psi,
                        cos_top1,
                        mask,
                        cos_topk_idx,
                        pool_k,
                        query_chunk,
                        db_tau,
                    )

            elif strategy == "naive_margin":

                def fn():
                    mask = margin >= float(naive_tau)
                    return _streaming_gated_top1(
                        method,
                        q_emb,
                        g_emb,
                        n_g,
                        psi,
                        cos_top1,
                        mask,
                        query_chunk,
                        gallery_chunk,
                        db_tau,
                    )

            elif strategy == "exact_global":

                def fn():
                    mask = margin >= (psi[cos_top1] - psi.min())
                    return _streaming_gated_top1(
                        method,
                        q_emb,
                        g_emb,
                        n_g,
                        psi,
                        cos_top1,
                        mask,
                        query_chunk,
                        gallery_chunk,
                        db_tau,
                    )

            elif strategy == "local_topk":

                def fn():
                    mask = _local_topk_margin_certified(
                        margin,
                        psi,
                        cos_top1,
                        cos_topk_idx,
                        strict=cert_local_strict,
                    )
                    return _streaming_gated_top1(
                        method,
                        q_emb,
                        g_emb,
                        n_g,
                        psi,
                        cos_top1,
                        mask,
                        query_chunk,
                        gallery_chunk,
                        db_tau,
                    )
            else:
                raise ValueError(strategy)

            _, t_e2e_s = _time_call(fn)

            top1_np = ref_top1.cpu().numpy()
            correct = _correct(top1_np, gt_map, qids_list)
            R1 = float(correct[has_gt].mean()) if has_gt.any() else 0.0
            gain = R1 - cos_R1
            if abs(full_gain) > 1e-9:
                gain_recovery_pct = 100.0 * gain / full_gain
            else:
                gain_recovery_pct = 100.0 if abs(gain) < 1e-9 else 0.0
            n_disagree = int((top1_np != full_top1_np).sum())
            n_q_tot = int(cert_mask.numel())
            pct_disagree_vs_full = 100.0 * float(n_disagree) / float(max(n_q_tot, 1))

            row_d: dict = {
                "method": method,
                "strategy": strategy,
                "selected_fraction": float(cert_mask.float().mean().item()),
                "n_selected": int(cert_mask.sum().item()),
                "n_total": n_q_tot,
                "cos_R1": cos_R1,
                "full_R1": full_R1,
                "strategy_R1": R1,
                "full_gain_R1": full_gain,
                "strategy_gain_R1": gain,
                "gain_recovery_pct": gain_recovery_pct,
                "top1_disagreements_vs_full": n_disagree,
                "pct_disagree_vs_full_gallery": pct_disagree_vs_full,
                "r1_gap_vs_full": float(full_R1 - R1),
                "time_e2e_ms": t_e2e_s * 1000,
            }
            if strategy == "selective_top50_pool_maxform":
                row_d["pct_disagree_vs_scalar_local"] = pct_dis_maxform_vs_scalar
            rows.append(row_d)

        full_t = next(
            x["time_e2e_ms"] for x in rows if x["method"] == method and x["strategy"] == "full"
        )
        pool_t = next(
            x["time_e2e_ms"]
            for x in rows
            if x["method"] == method and x["strategy"] == "rerank_cos_topk_pool"
        )
        for x in rows:
            if x["method"] == method:
                te = max(float(x["time_e2e_ms"]), 1e-9)
                suv_full = float(full_t) / te
                suv_pool = float(pool_t) / te
                x["speedup_vs_full_gallery"] = suv_full
                x["speedup_vs_rerank_cos_topk_pool"] = suv_pool
                x["speedup_e2e"] = suv_full

    out_cell = {
        "cell": f"{backbone}|{direction}|{','.join(gallery_splits)}",
        "n_queries": n_q,
        "n_gallery": n_g,
        "n_queries_with_gt": int(has_gt.sum()),
        "cosine_R1": cos_R1,
        "cert_local_strict": bool(cert_local_strict),
        "rows": rows,
        "streaming": True,
    }
    if skip_rates:
        out_cell["quantile_skip_margin_vs_psi"] = quantile_skip_margin_vs_psi
        out_cell["quantile_skip_cert_local_topk"] = min(topk_cand, int(cos_topk_idx.shape[1]))
    return out_cell


# ===================================================================== #
# Main                                                                   #
# ===================================================================== #


def evaluate_cell(
    dataset: str,
    backbone: str,
    direction: str,
    query_split: str,
    gallery_splits: list[str],
    csls_k: int,
    qb_tau: float,
    db_tau: float,
    nnn_k: int,
    nnn_w: float,
    naive_tau: float,
    topk_cand: int,
    rerank_pool_k: int,
    batch_size: int,
    max_sims_gb: float,
    query_chunk: int,
    qb_psi_query_chunk: int,
    skip_rates: list[float],
    device: torch.device,
    *,
    cert_local_strict: bool = False,
) -> dict:
    print(f"  loading embeddings ...", flush=True)
    q_emb, g_emb, q_ids, g_ids = load_laion_embeddings(
        dataset=dataset,
        backbone=backbone,
        direction=direction,
        query_split=query_split,
        gallery_splits=gallery_splits,
        device=device,
    )
    n_q, n_g = q_emb.shape[0], g_emb.shape[0]
    print(f"  queries={n_q}  gallery={n_g}", flush=True)

    sims_bytes = n_q * n_g * 4
    limit_bytes = max_sims_gb * (1024**3)
    if sims_bytes > limit_bytes:
        print(
            f"  sims would be {sims_bytes / (1024**3):.2f} GiB > --max-sims-gb {max_sims_gb}; using streaming",
            flush=True,
        )
        return evaluate_cell_streaming(
            q_emb,
            g_emb,
            q_ids,
            g_ids,
            backbone,
            direction,
            gallery_splits,
            csls_k,
            qb_tau,
            db_tau,
            nnn_k,
            nnn_w,
            naive_tau,
            topk_cand,
            gallery_chunk=batch_size,
            query_chunk=query_chunk,
            qb_psi_query_chunk=qb_psi_query_chunk,
            skip_rates=skip_rates,
            rerank_pool_k=min(int(rerank_pool_k), int(topk_cand)),
            device=device,
            cert_local_strict=cert_local_strict,
        )

    qids_list = [str(x) for x in q_ids]
    gt_map = build_pair_gt_mapping(q_ids, g_ids, direction)
    has_gt = np.array([bool(gt_map.get(q)) for q in qids_list], dtype=bool)
    print(f"  queries_with_gt={int(has_gt.sum())}", flush=True)

    # sims (chunked compute keeps peak memory bounded)
    print(f"  computing sims (chunked) ...", flush=True)
    t0 = time.perf_counter()
    sims = compute_sims_full(q_emb, g_emb, batch_size=batch_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_sims = time.perf_counter() - t0
    print(f"  sims done in {t_sims:.2f}s, shape={tuple(sims.shape)}", flush=True)

    # cosine top-1, margin, top-K (used by the local cert)
    cos_top1, margin = cos_top1_margin_batched(sims, batch_size=batch_size)
    cos_topk_idx = cos_topk_batched(sims, k=topk_cand, batch_size=batch_size)
    cos_top1_np = cos_top1.cpu().numpy()
    cos_correct = _correct(cos_top1_np, gt_map, qids_list)
    cos_R1 = float(cos_correct[has_gt].mean()) if has_gt.any() else 0.0
    print(f"  cos R@1 = {cos_R1:.4f}", flush=True)

    method_kw_common = dict(
        csls_k=csls_k,
        qb_tau=qb_tau,
        db_tau=db_tau,
        nnn_k=nnn_k,
        nnn_w=nnn_w,
        qb_psi_query_chunk=qb_psi_query_chunk,
        rerank_pool_k=min(int(rerank_pool_k), int(topk_cand)),
    )
    pool_k = min(int(rerank_pool_k), int(topk_cand))

    rows: list[dict] = []
    quantile_skip_margin_vs_psi: dict[str, list[dict]] = {}
    for method in METHODS:
        print(f"  method = {method}", flush=True)
        psi = _build_psi(method, sims, **method_kw_common)

        cos_topk_sims_all = _dense_cos_topk_sims(sims, cos_topk_idx)
        psi_pack_method = psi[cos_topk_idx]
        cert_scalar_mask_dense = _local_topk_margin_certified(
            margin,
            psi,
            cos_top1,
            cos_topk_idx,
            strict=cert_local_strict,
        )
        scalar_pool_top1 = _gated_top1_pool(
            method,
            sims,
            psi,
            cos_top1,
            cert_scalar_mask_dense,
            cos_topk_idx,
            pool_k,
            **method_kw_common,
        )
        cert_maxform_mask_dense = _local_topk_maxform_certified(
            cos_topk_sims_all, psi_pack_method, strict=cert_local_strict
        )
        maxform_pool_top1 = _gated_top1_pool(
            method,
            sims,
            psi,
            cos_top1,
            cert_maxform_mask_dense,
            cos_topk_idx,
            pool_k,
            **method_kw_common,
        )
        pct_dis_maxform_vs_scalar_dense = float(
            100.0 * float((maxform_pool_top1 != scalar_pool_top1).float().mean().item())
        )

        # Reference: full method R@1 (also serves as recovery denominator)
        full_top1_t = _full_top1(method, sims, psi, **method_kw_common)
        full_top1_np = full_top1_t.cpu().numpy()
        full_correct = _correct(full_top1_np, gt_map, qids_list)
        full_R1 = float(full_correct[has_gt].mean()) if has_gt.any() else 0.0
        full_gain = full_R1 - cos_R1

        if skip_rates:
            k_cert = min(topk_cand, int(cos_topk_idx.shape[1]))
            quantile_skip_margin_vs_psi[method] = compute_margin_vs_psi_quantile_curve(
                psi,
                cos_top1,
                margin,
                cos_topk_idx,
                full_top1_t,
                k_cert,
                skip_rates,
                cert_local_strict=cert_local_strict,
            )

        for strategy in STRATEGIES:
            r = evaluate_strategy(
                method=method,
                strategy=strategy,
                sims=sims,
                psi=psi,
                cos_top1=cos_top1,
                margin=margin,
                cos_topk_idx=cos_topk_idx,
                naive_tau=naive_tau,
                cert_local_strict=cert_local_strict,
                **method_kw_common,
            )
            top1_np = r.pop("top1_ref").cpu().numpy()
            correct = _correct(top1_np, gt_map, qids_list)
            R1 = float(correct[has_gt].mean()) if has_gt.any() else 0.0
            gain = R1 - cos_R1
            if abs(full_gain) > 1e-9:
                gain_recovery_pct = 100.0 * gain / full_gain
            else:
                gain_recovery_pct = 100.0 if abs(gain) < 1e-9 else 0.0

            # Top-1 disagreements vs full (sanity / soundness)
            n_disagree = int((top1_np != full_top1_np).sum())
            n_q_tot = int(margin.shape[0])
            pct_disagree_vs_full = 100.0 * float(n_disagree) / float(max(n_q_tot, 1))

            r.update(
                {
                    "cos_R1": cos_R1,
                    "full_R1": full_R1,
                    "strategy_R1": R1,
                    "full_gain_R1": full_gain,
                    "strategy_gain_R1": gain,
                    "gain_recovery_pct": gain_recovery_pct,
                    "top1_disagreements_vs_full": n_disagree,
                    "pct_disagree_vs_full_gallery": pct_disagree_vs_full,
                    "r1_gap_vs_full": float(full_R1 - R1),
                }
            )
            if strategy == "selective_top50_pool_maxform":
                r["pct_disagree_vs_scalar_local"] = pct_dis_maxform_vs_scalar_dense
            rows.append(r)

        # speedup vs full-gallery and vs always-rerank cosine-top-K pool (same method block)
        full_t = next(
            x["time_e2e_ms"] for x in rows if x["method"] == method and x["strategy"] == "full"
        )
        pool_t = next(
            x["time_e2e_ms"]
            for x in rows
            if x["method"] == method and x["strategy"] == "rerank_cos_topk_pool"
        )
        for x in rows:
            if x["method"] == method:
                te = max(float(x["time_e2e_ms"]), 1e-9)
                suv_full = float(full_t) / te
                suv_pool = float(pool_t) / te
                x["speedup_vs_full_gallery"] = suv_full
                x["speedup_vs_rerank_cos_topk_pool"] = suv_pool
                x["speedup_e2e"] = suv_full

    out_dense = {
        "cell": f"{backbone}|{direction}|{','.join(gallery_splits)}",
        "n_queries": n_q,
        "n_gallery": n_g,
        "n_queries_with_gt": int(has_gt.sum()),
        "cosine_R1": cos_R1,
        "cert_local_strict": bool(cert_local_strict),
        "rows": rows,
        "streaming": False,
    }
    if skip_rates:
        out_dense["quantile_skip_margin_vs_psi"] = quantile_skip_margin_vs_psi
        out_dense["quantile_skip_cert_local_topk"] = min(topk_cand, int(cos_topk_idx.shape[1]))
    return out_dense


def write_md(path: Path, cells: list[dict], cfg: dict) -> None:
    lines = [
        "# LAION Gating-Strategy Comparison",
        "",
        f"Dataset: `{cfg['dataset']}` | backbone: `{cfg['backbone']}` | "
        f"naive_tau: `{cfg['naive_tau']}` | topk_cand: `{cfg['topk_cand']}` | "
        f"rerank_pool_k: `{cfg['rerank_pool_k']}` | "
        f"CSLS k: `{cfg['csls_k']}` | QB tau: `{cfg['qb_tau']}` | "
        f"DB tau: `{cfg['db_tau']}` | NNN: `k={cfg['nnn_k']}, w={cfg['nnn_w']}` | "
        f"max_sims_gb: `{cfg['max_sims_gb']}` | query_chunk: `{cfg['query_chunk']}` | "
        f"qb_psi_q_chunk: `{cfg['qb_psi_query_chunk']}` | "
        f"skip_rates (ψ vs margin quantiles): `{cfg.get('skip_rates', [])}` | "
        f"cert_local_strict (scalar/global strict `>`; max-form strict `< 0`): `{cfg.get('cert_local_strict', False)}`",
        "",
        "Strategies:",
        "- `full`: baseline (rerank every query over full gallery).",
        "- `rerank_cos_topk_pool`: always rerank, scores restricted to cosine top-K pool (`rerank_pool_k`).",
        "- `selective_top50_pool`: **same skip as local_topk**; when reranking, scores restricted to "
        "cosine top-K pool (`rerank_pool_k`, default 50) instead of full gallery.",
        "- `selective_top50_pool_exact_global`: **same pool rerank**, skip rule uses gallery-global "
        "`ψ(c1*) − min_gallery ψ` with strict `>` when `--cert-local-strict`.",
        "- `selective_top50_pool_maxform`: **same pool rerank**, Prop. 2 candidate-local max-form on "
        "the cosine shortlist (`max_{c≠c1∈C_K} [ψ(c1)−ψ(c)−(s(q,c1)−s(q,c))] < 0` with strict `<`).",
        "- `naive_margin`: skip rerank if `m(q) >= naive_tau`.",
        "- `exact_global`: skip rerank if `m(q) >= psi(c1*) - min_c psi(c)` (sound).",
        "- `local_topk`: skip rerank if `m(q) >= psi(c1*) - min_{c in topK(q)} psi(c)` "
        "(or **>** when `cert_local_strict`).",
        "  (NOT sound; surrogate that ignores outside-topK threats).",
        "",
        "`selected_fraction`: queries gated to cosine top-1 (skipping rerank).",
        "`gain recovery`: `(R@1_strategy - cos_R1) / (R@1_full - cos_R1)`.",
        "`top1 disagree vs full` / `% disagree`: per-query top-1 mismatches vs full gallery.",
        "`r1_gap_vs_full`: `R@1_full − R@1_strategy` (penalty vs exhaustive rerank).",
        "`pct_disagree_vs_scalar_local`: only on `selective_top50_pool_maxform` — fraction of queries "
        "whose gated top-1 differs from candidate-local scalar selective pooling.",
        "`speedup vs full` / `speedup vs pool`: wall-clock ratios vs `full` and vs `rerank_cos_topk_pool`.",
        "",
    ]
    for cell in cells:
        lines += [
            f"## {cell['cell']}",
            "",
            f"- queries: `{cell['n_queries']}` | gallery: `{cell['n_gallery']}` | "
            f"with GT: `{cell['n_queries_with_gt']}` | streaming: `{cell.get('streaming', False)}`",
            f"- cosine R@1: `{cell['cosine_R1']:.4f}`",
            "",
            "| Method | Strategy | %selected | strat R@1 | ΔR@1 vs cos | gain recovery | "
            "disagree vs full | %dis vs full | spd vs full | spd vs pool | wall ms |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in cell["rows"]:
            lines.append(
                f"| {r['method']} | `{r['strategy']}` | "
                f"{r['selected_fraction']*100:.2f}% | "
                f"{r['strategy_R1']:.4f} | "
                f"{r['strategy_gain_R1']:+.4f} | "
                f"{r['gain_recovery_pct']:.2f}% | "
                f"{r['top1_disagreements_vs_full']} | "
                f"{r.get('pct_disagree_vs_full_gallery', 0.0):.2f}% | "
                f"{r.get('speedup_vs_full_gallery', r.get('speedup_e2e', 1.0)):.2f}x | "
                f"{r.get('speedup_vs_rerank_cos_topk_pool', 1.0):.2f}x | "
                f"{r['time_e2e_ms']:.2f} |"
            )
        cert_keys = (
            "selective_top50_pool_exact_global",
            "selective_top50_pool",
            "selective_top50_pool_maxform",
        )
        cert_labels = {
            "selective_top50_pool_exact_global": "global ψ_min (pool rerank)",
            "selective_top50_pool": "candidate-local scalar (pool rerank)",
            "selective_top50_pool_maxform": "candidate-local max-form Prop.2 (pool rerank)",
        }
        lines += [
            "",
            "### Certificate forms (global vs local scalar vs Prop. 2 max-form)",
            "",
            "Same reranking protocol as `selective_top50_pool` (cosine top-`rerank_pool_k` pool only when not skipping). "
            "`cert_local_strict` controls strict `>` / `<` for global and scalar margin rules and strict `< 0` for max-form.",
            "",
            "| Method | Certificate | %skip | R@1 | R@1 gap vs full | %dis vs full | %dis maxform vs scalar | spd vs full | spd vs pool |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        methods_order = ("CSLS", "QB-Norm", "DB-Norm", "NNN")
        for m in methods_order:
            for strat in cert_keys:
                hit = next(
                    (x for x in cell["rows"] if x["method"] == m and x["strategy"] == strat),
                    None,
                )
                if hit is None:
                    continue
                r = hit
                disc_sc = r.get("pct_disagree_vs_scalar_local")
                disc_sc_s = f"{disc_sc:.2f}" if disc_sc is not None else "—"
                lines.append(
                    f"| {r['method']} | {cert_labels[r['strategy']]} | "
                    f"{r['selected_fraction']*100:.2f}% | "
                    f"{r['strategy_R1']:.4f} | "
                    f"{r.get('r1_gap_vs_full', 0.0):+.4f} | "
                    f"{r.get('pct_disagree_vs_full_gallery', 0.0):.2f}% | "
                    f"{disc_sc_s} | "
                    f"{r.get('speedup_vs_full_gallery', r.get('speedup_e2e', 1.0)):.2f}x | "
                    f"{r.get('speedup_vs_rerank_cos_topk_pool', 1.0):.2f}x |"
                )
        lines += [
            "",
            "### Wall-clock (mean over 5 reps after warmup, ms)",
            "",
            "| Method | full | pool | sel_scalar | sel_global | sel_maxform | naive_m | exact_gl | local_topk |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        # Pivot for timing table
        timing_by_method: dict[str, dict[str, float]] = {}
        for r in cell["rows"]:
            timing_by_method.setdefault(r["method"], {})[r["strategy"]] = r["time_e2e_ms"]
        for m, d in timing_by_method.items():
            lines.append(
                f"| {m} | "
                f"{d.get('full', 0.0):.2f} | "
                f"{d.get('rerank_cos_topk_pool', 0.0):.2f} | "
                f"{d.get('selective_top50_pool', 0.0):.2f} | "
                f"{d.get('selective_top50_pool_exact_global', 0.0):.2f} | "
                f"{d.get('selective_top50_pool_maxform', 0.0):.2f} | "
                f"{d.get('naive_margin', 0.0):.2f} | "
                f"{d.get('exact_global', 0.0):.2f} | "
                f"{d.get('local_topk', 0.0):.2f} |"
            )
        lines.append("")
        qcurve = cell.get("quantile_skip_margin_vs_psi")
        if qcurve:
            lines += [
                "### Matched-skip-rate margin vs ψ-slack (disagreement vs full rerank)",
                "",
                "Same semantics as `experiments/compare_margin_vs_psi_gate.py`: skip → cosine top-1 at matched workload; "
                "τ_m = quantile_{1−s}(margin), δ_sl = quantile_{1−s}(slack_local) with candidate-local min ψ over cosine top-K.",
                "",
                f"- `quantile_skip_cert_local_topk`: `{cell.get('quantile_skip_cert_local_topk', '')}`",
                "",
            ]
            for meth, qrows in qcurve.items():
                lines.append(f"#### {meth}")
                lines.append("")
                lines.append(
                    "| target skip | τ_margin | δ_slack | skip_m | skip_ψ | "
                    "dis_m % | dis_ψ % | Δ_pp ψ−m |"
                )
                lines.append(
                    "|-------------|---------:|--------:|-------:|-------:|---------:|--------:|---------:|"
                )
                for qr in qrows:
                    lines.append(
                        f"| {100*qr['skip_rate_target']:.1f}% | {qr['tau_margin']:.5f} | "
                        f"{qr['delta_slack_local']:.5f} | "
                        f"{100*qr['skip_rate_empirical_margin']:.2f} | "
                        f"{100*qr['skip_rate_empirical_psi']:.2f} | "
                        f"{100*qr['pct_disagree_with_full_margin_gate']:.2f} | "
                        f"{100*qr['pct_disagree_with_full_psi_gate']:.2f} | "
                        f"{qr['delta_disagree_psi_minus_margin_pp']:+.2f} |"
                    )
                lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="laion_sample")
    ap.add_argument("--backbone", type=str, default="laion")
    ap.add_argument("--direction", type=str, default="t2i", choices=["i2t", "t2i"])
    ap.add_argument("--query-split", type=str, default="test")
    ap.add_argument(
        "--gallery-splits",
        type=str,
        default="test",
        help="Comma-separated splits to concatenate as gallery, e.g. 'test' (default) "
        "or 'test,val,train' for the full LAION-543K gallery.",
    )
    ap.add_argument("--csls-k", type=int, default=20)
    ap.add_argument("--qb-tau", type=float, default=20.0)
    ap.add_argument("--db-tau", type=float, default=20.0)
    ap.add_argument("--nnn-k", type=int, default=64)
    ap.add_argument("--nnn-w", type=float, default=0.5)
    ap.add_argument("--naive-tau", type=float, default=0.05)
    ap.add_argument("--topk-cand", type=int, default=50)
    ap.add_argument(
        "--rerank-pool-k",
        type=int,
        default=50,
        help="Cosine pool size for `rerank_cos_topk_pool` and `selective_top50_pool` reranking "
        "(capped by --topk-cand).",
    )
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument(
        "--max-sims-gb",
        type=float,
        default=14.0,
        help="If N_queries×N_gallery×4 bytes exceeds this, run streaming matmul (required for ~543K gallery).",
    )
    ap.add_argument(
        "--query-chunk",
        type=int,
        default=512,
        help="Query batch size for streaming cosine stats / top-1 passes.",
    )
    ap.add_argument(
        "--qb-psi-query-chunk",
        type=int,
        default=2048,
        help="Query chunk size when reducing logsumexp for QB/DB psi (dense and streaming).",
    )
    ap.add_argument(
        "--skip-rates",
        type=str,
        default="0.15,0.25,0.35,0.45,0.55,0.60,0.62,0.65,0.69,0.71",
        help="Comma-separated target skip fractions for matched-quantile margin vs ψ-slack curve "
        "(same as compare_margin_vs_psi_gate). Includes higher skips for finer curves. "
        "Empty string disables.",
    )
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument(
        "--cert-local-strict",
        action="store_true",
        help="For local_topk / selective_top50_pool / ψ-slack quantile curve: require strict "
        "m(q) > B_q^local (equivalently slack > 0) instead of ≥.",
    )
    ap.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="Optional suffix appended to output stem, e.g. `selective` → ..._test_val_train_selective.json",
    )
    args = ap.parse_args()

    skip_rates = [float(x.strip()) for x in args.skip_rates.split(",") if x.strip()]

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    gallery_splits = [x.strip() for x in args.gallery_splits.split(",") if x.strip()]

    print(f"[cell] {args.backbone} | {args.direction} | gallery={gallery_splits}", flush=True)
    cell = evaluate_cell(
        dataset=args.dataset,
        backbone=args.backbone,
        direction=args.direction,
        query_split=args.query_split,
        gallery_splits=gallery_splits,
        csls_k=args.csls_k,
        qb_tau=args.qb_tau,
        db_tau=args.db_tau,
        nnn_k=args.nnn_k,
        nnn_w=args.nnn_w,
        naive_tau=args.naive_tau,
        topk_cand=args.topk_cand,
        rerank_pool_k=args.rerank_pool_k,
        batch_size=args.batch_size,
        max_sims_gb=args.max_sims_gb,
        query_chunk=args.query_chunk,
        qb_psi_query_chunk=args.qb_psi_query_chunk,
        skip_rates=skip_rates,
        device=device,
        cert_local_strict=args.cert_local_strict,
    )

    cfg = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "direction": args.direction,
        "query_split": args.query_split,
        "gallery_splits": gallery_splits,
        "csls_k": args.csls_k,
        "qb_tau": args.qb_tau,
        "db_tau": args.db_tau,
        "nnn_k": args.nnn_k,
        "nnn_w": args.nnn_w,
        "naive_tau": args.naive_tau,
        "topk_cand": args.topk_cand,
        "rerank_pool_k": args.rerank_pool_k,
        "batch_size": args.batch_size,
        "max_sims_gb": args.max_sims_gb,
        "query_chunk": args.query_chunk,
        "qb_psi_query_chunk": args.qb_psi_query_chunk,
        "skip_rates": skip_rates,
        "cert_local_strict": bool(args.cert_local_strict),
        "output_tag": str(args.output_tag).strip(),
        "generated": datetime.now().isoformat(),
    }
    out = {"config": cfg, "cells": [cell]}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(gallery_splits)
    stem = f"LaionGating_{args.dataset}_{args.direction}_{suffix}"
    tag = str(args.output_tag).strip()
    if tag:
        stem = f"{stem}_{tag}"
    js = OUTPUT_DIR / f"{stem}.json"
    md = OUTPUT_DIR / f"{stem}.md"
    js.write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_md(md, [cell], cfg)
    print(f"Wrote:\n  {js}\n  {md}")


if __name__ == "__main__":
    main()
