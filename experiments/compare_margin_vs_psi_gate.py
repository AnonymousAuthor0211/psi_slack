#!/usr/bin/env python3
"""
Compare naive margin vs candidate-local ψ-slack gate **at matched skip rates**, with optional
**label-free logistic gate** (train split fits P(cosine agrees with full top-1 | margin, slack)).

Semantics: **skip full rerank** → cosine top-1; else **full-method top-1**.

Skip-rate matching:
  - **Margin**: skip iff m(q) ≥ τ = quantile_{1-s}(m).
  - **ψ**: skip iff slack_local ≥ δ = quantile_{1-s}(slack),
    slack_local = m − (ψ(c1*) − min_{c∈cos-top-K} ψ(c)).
  - **Learned** (optional): logistic on [margin, slack] predicting y=1[cos_top1==full_top1];
    skip iff P(y=1) ≥ quantile_{1-s}(P) on the **eval** split (`--learned-logistic` evaluates
    margin/ψ/learned on the same eval queries only).

Benchmarks: any `embeddings_<backbone>/<dataset>_test_*.npz` layout (not COCO-only).
**LAION**: `--loader laion` plus `--max-queries` / `--max-gallery` to bound memory. For LAION-scale
**gain-recovery / speedup Pareto**, use `experiments/laion_gating_comparison.py`.

Outputs: evaluation_results/tables_GPU/MarginVsPsiGate_<dataset>.{json,md}

Usage:
  CUDA_VISIBLE_DEVICES=4 python experiments/compare_margin_vs_psi_gate.py \\
      --dataset coco_captions --backbones clip --directions i2t,t2i \\
      --cert-local-topk 50 --skip-rates 0.15,0.25,0.35,0.45

  python experiments/compare_margin_vs_psi_gate.py ... --learned-logistic --learned-train-frac 0.5

  python experiments/compare_margin_vs_psi_gate.py --loader laion --dataset laion_sample \\
      --laion-query-split test --laion-gallery-splits test --max-queries 4096 --max-gallery 8192
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from psi_slack_pkg.embeddings import load_embeddings  # noqa: E402
from psi_slack_pkg.noop_certificate_speedup import (  # noqa: E402
    _build_psi,
    _cos_top1_and_margin,
    _full_top1,
)

OUTPUT_DIR = project_root / "evaluation_results" / "tables_GPU"

METHODS = ("CSLS", "QB-Norm", "DB-Norm", "NNN")

SCOPE_NOTE = {
    "what_this_script_compares": (
        "Naive cosine-margin thresholds vs candidate-local ψ-slack vs optional label-free "
        "logistic on (margin, slack). Disagreement is measured against full-method top-1."
    ),
    "learned_baseline": (
        "Logistic regression predicts P(cos_top1 == full_top1) without GT; trained on a "
        "random train split, thresholds chosen on the eval split at quantile (1−s). "
        "This is a lightweight sanity baseline—not a tuned neural confidence model."
    ),
    "laion_pareto_pointer": (
        "For LAION at scale (large gallery / streaming), run "
        "`experiments/laion_gating_comparison.py` with `--gallery-splits test,val,train` and "
        "`--skip-rates ...`: it reports gain-recovery Pareto *and* the matched-quantile "
        "margin vs ψ-slack disagreement JSON (`quantile_skip_margin_vs_psi`). "
        "Tiny `--max-queries`/`--max-gallery` subsets often yield 0% disagreement at aggressive skips."
    ),
}


def _split_train_eval(n: int, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = max(1, min(n - 1, int(round(n * train_frac))))
    tr = perm[:n_train]
    ev = perm[n_train:]
    if len(ev) == 0:
        ev = tr[-1:]
        tr = tr[:-1]
    return np.sort(tr), np.sort(ev)


def _load_embeddings_standard(
    dataset: str,
    direction: str,
    backbone: str,
    device: torch.device,
    max_queries: int,
    max_gallery: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_emb, g_emb, _, _ = load_embeddings(dataset, direction, backbone)
    q_emb = F.normalize(q_emb.to(device), dim=1)
    g_emb = F.normalize(g_emb.to(device), dim=1)
    if max_queries > 0:
        q_emb = q_emb[:max_queries]
    if max_gallery > 0:
        g_emb = g_emb[:max_gallery]
    return q_emb, g_emb


def _load_embeddings_laion(
    dataset: str,
    backbone: str,
    direction: str,
    device: torch.device,
    query_split: str,
    gallery_splits: list[str],
    max_queries: int,
    max_gallery: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from psi_slack_pkg.laion import load_laion_embeddings  # noqa: E402

    q_emb, g_emb, _, _ = load_laion_embeddings(
        dataset, backbone, direction, query_split, gallery_splits, device
    )
    if max_queries > 0:
        q_emb = q_emb[:max_queries]
    if max_gallery > 0:
        g_emb = g_emb[:max_gallery]
    return q_emb, g_emb


def _slack_local(
    psi: torch.Tensor,
    cos_top1: torch.Tensor,
    margin: torch.Tensor,
    cos_topk_idx: torch.Tensor,
    k_local: int,
) -> torch.Tensor:
    psi_c1 = psi[cos_top1]
    k_eff = min(k_local, cos_topk_idx.shape[1])
    idx = cos_topk_idx[:, :k_eff]
    psi_min_q = psi[idx].min(dim=1).values
    b_q = psi_c1 - psi_min_q
    return margin - b_q


def _run_method_metrics(
    method: str,
    sims: torch.Tensor,
    cos_top1: torch.Tensor,
    margin: torch.Tensor,
    psi: torch.Tensor,
    cos_topk_idx: torch.Tensor,
    k_local: int,
    skip_rates: list[float],
    train_idx: np.ndarray | None = None,
    eval_idx: np.ndarray | None = None,
    learned_logistic: bool = False,
    **kw,
) -> tuple[list[dict], dict | None]:
    full_top1 = _full_top1(method, sims, psi, **kw)
    slack = _slack_local(psi, cos_top1, margin, cos_topk_idx, k_local)

    margin_np = margin.detach().cpu().numpy().astype(np.float64)
    slack_np = slack.detach().cpu().numpy().astype(np.float64)
    cos_np = cos_top1.cpu().numpy().astype(np.int64)
    full_np = full_top1.cpu().numpy().astype(np.int64)

    learned_meta: dict | None = None
    p_ev: np.ndarray | None = None

    if learned_logistic and train_idx is not None and eval_idx is not None:
        from sklearn.linear_model import LogisticRegression  # noqa: E402

        X_tr = np.stack([margin_np[train_idx], slack_np[train_idx]], axis=1)
        y_tr = (cos_np[train_idx] == full_np[train_idx]).astype(np.int64)
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        )
        clf.fit(X_tr, y_tr)
        X_ev = np.stack([margin_np[eval_idx], slack_np[eval_idx]], axis=1)
        p_ev = clf.predict_proba(X_ev)[:, 1].astype(np.float64)
        learned_meta = {
            "coef_margin_slack": [float(clf.coef_[0, 0]), float(clf.coef_[0, 1])],
            "intercept": float(clf.intercept_[0]),
            "n_train": int(train_idx.shape[0]),
            "n_eval": int(eval_idx.shape[0]),
        }
        margin_np = margin_np[eval_idx]
        slack_np = slack_np[eval_idx]
        cos_np = cos_np[eval_idx]
        full_np = full_np[eval_idx]
    else:
        margin_np = np.ascontiguousarray(margin_np)
        slack_np = np.ascontiguousarray(slack_np)

    n = margin_np.shape[0]

    rows = []
    for s in skip_rates:
        if not (0.0 < s < 1.0):
            continue
        tau = float(np.quantile(margin_np, 1.0 - s))
        delta = float(np.quantile(slack_np, 1.0 - s))

        skip_m = margin_np >= tau
        skip_psi = slack_np >= delta
        sr_m = float(np.mean(skip_m))
        sr_psi = float(np.mean(skip_psi))

        gated_m = np.where(skip_m, cos_np, full_np)
        gated_psi = np.where(skip_psi, cos_np, full_np)

        disagree_m = float(np.mean(gated_m != full_np))
        disagree_psi = float(np.mean(gated_psi != full_np))

        cos_ne_full = cos_np != full_np
        among_skip_m = float(np.mean(cos_ne_full[skip_m])) if skip_m.any() else 0.0
        among_skip_psi = float(np.mean(cos_ne_full[skip_psi])) if skip_psi.any() else 0.0

        row: dict = {
            "skip_rate_target": float(s),
            "tau_margin": tau,
            "delta_slack_local": delta,
            "skip_rate_empirical_margin": sr_m,
            "skip_rate_empirical_psi": sr_psi,
            "pct_disagree_with_full_margin_gate": disagree_m,
            "pct_disagree_with_full_psi_gate": disagree_psi,
            "delta_disagree_psi_minus_margin_pp": float(
                100.0 * (disagree_m - disagree_psi)
            ),
            "among_skipped_pct_cos_neq_full_margin": among_skip_m,
            "among_skipped_pct_cos_neq_full_psi": among_skip_psi,
            "n_queries": int(n),
        }

        if p_ev is not None:
            thr_p = float(np.quantile(p_ev, 1.0 - s))
            skip_l = p_ev >= thr_p
            sr_l = float(np.mean(skip_l))
            gated_l = np.where(skip_l, cos_np, full_np)
            disagree_l = float(np.mean(gated_l != full_np))
            among_skip_l = (
                float(np.mean(cos_ne_full[skip_l])) if skip_l.any() else 0.0
            )
            row["tau_learned_prob"] = thr_p
            row["skip_rate_empirical_learned"] = sr_l
            row["pct_disagree_with_full_learned_gate"] = disagree_l
            row["delta_pp_learned_minus_psi"] = float(
                100.0 * (disagree_l - disagree_psi)
            )
            row["among_skipped_pct_cos_neq_full_learned"] = among_skip_l

        rows.append(row)

    return rows, learned_meta


def evaluate_cell(
    dataset: str,
    direction: str,
    backbone: str,
    device: torch.device,
    cert_local_topk: int,
    skip_rates: list[float],
    csls_k: int,
    qb_tau: float,
    db_tau: float,
    nnn_k: int,
    nnn_w: float,
    loader: str,
    max_queries: int,
    max_gallery: int,
    laion_query_split: str,
    laion_gallery_splits: list[str],
    learned_logistic: bool,
    learned_train_frac: float,
    learned_seed: int,
) -> dict:
    if loader == "standard":
        q_emb, g_emb = _load_embeddings_standard(
            dataset, direction, backbone, device, max_queries, max_gallery
        )
    elif loader == "laion":
        q_emb, g_emb = _load_embeddings_laion(
            dataset,
            backbone,
            direction,
            device,
            laion_query_split,
            laion_gallery_splits,
            max_queries,
            max_gallery,
        )
    else:
        raise ValueError(f"Unknown --loader {loader}")

    sims = q_emb @ g_emb.T
    cos_top1, margin = _cos_top1_and_margin(sims)

    k_local = min(cert_local_topk, sims.shape[1])
    cos_topk_idx = sims.topk(k_local, dim=1).indices

    kw = {
        "csls_k": csls_k,
        "qb_tau": qb_tau,
        "db_tau": db_tau,
        "nnn_k": nnn_k,
        "nnn_w": nnn_w,
    }

    n_q = int(sims.shape[0])
    train_idx = eval_idx = None
    if learned_logistic:
        train_idx, eval_idx = _split_train_eval(n_q, learned_train_frac, learned_seed)

    methods_out: dict = {}
    learned_calibration: dict = {}

    for method in METHODS:
        psi = _build_psi(method, sims, **kw)
        rows, lmeta = _run_method_metrics(
            method,
            sims,
            cos_top1,
            margin,
            psi,
            cos_topk_idx,
            cert_local_topk,
            skip_rates,
            train_idx=train_idx,
            eval_idx=eval_idx,
            learned_logistic=learned_logistic,
            **kw,
        )
        methods_out[method] = rows
        if lmeta is not None:
            learned_calibration[method] = lmeta

    cell: dict = {
        "cell": f"{backbone}|{direction}",
        "loader": loader,
        "cert_local_topk": k_local,
        "methods": methods_out,
    }
    if learned_logistic:
        cell["learned_split"] = {
            "train_frac": learned_train_frac,
            "seed": learned_seed,
            "calibration": learned_calibration,
        }
    return cell


def write_md(path: Path, dataset: str, cells: list[dict], cfg: dict) -> None:
    learned_on = bool(cfg.get("learned_logistic"))
    lines = [
        "# Margin vs ψ-slack vs optional learned logistic (matched skip rate)",
        "",
        f"Dataset: `{dataset}` | loader: `{cfg.get('loader', 'standard')}` | "
        f"cosine top-K for local ψ: `{cfg['cert_local_topk']}`",
        "",
        "**Margin gate**: skip full rerank iff $m(q) \\geq \\tau$, with $\\tau$ set to the ",
        "$(1-s)$-quantile of $m$ so the empirical skip rate ≈ $s$.",
        "",
        "**ψ gate**: skip iff $\\mathrm{slack} \\geq \\delta$, ",
        "$\\mathrm{slack}=m-(\\psi(c_1^*)-\\min_{c\\in\\mathrm{top\\text{-}K}}\\psi(c))$, ",
        "$\\delta$ from the $(1-s)$-quantile of slack (same target skip $s$).",
        "",
    ]
    if learned_on:
        lines += [
            "**Learned gate** (eval split): logistic regression on "
            "$(m, \\mathrm{slack})$ to predict $\\mathbb{1}[\\mathrm{cos\\text{-}top\\text{-}1}=\\mathrm{full\\text{-}top\\text{-}1}]$, "
            "trained on a random train split; skip iff $\\hat p \\geq \\tau_p$ at the $(1-s)$ quantile of $\\hat p$ **on eval**.",
            "",
        ]
    lines += [
        "**pct_disagree***: fraction of queries where gated top-1 $\\neq$ full rerank top-1.",
        "**Δ_pp**: $100\\times(\\mathrm{dis}_{\\mathrm{margin}}-\\mathrm{dis}_{\\psi})$ — positive means ψ gate disagrees less.",
        "",
        "> See JSON field `scope_note` for benchmarks beyond COCO and pointers to LAION-scale Pareto tooling.",
        "",
    ]
    for c in cells:
        lines.append(f"## {c['cell']}")
        lines.append("")
        for method, rows in c["methods"].items():
            lines.append(f"### {method}")
            lines.append("")
            sample = rows[0] if rows else {}
            has_learned = "pct_disagree_with_full_learned_gate" in sample
            if has_learned:
                lines.append(
                    "| target skip | τ_margin | δ_slack | τ_p | skip_m | skip_ψ | skip_L | "
                    "dis_m % | dis_ψ % | dis_L % | Δ_pp ψ−m | Δ_pp L−ψ | "
                    "among_skip cos≠full_m | among_skip cos≠full_ψ | among_skip cos≠full_L |"
                )
                lines.append(
                    "|-------------|---------:|--------:|----:|-------:|-------:|-------:|"
                    "---------:|--------:|--------:|-----------:|-----------:|"
                    "----------------------:|----------------------:|----------------------:|"
                )
            else:
                lines.append(
                    "| target skip | τ_margin | δ_slack | skip_m | skip_ψ | "
                    "dis_m % | dis_ψ % | Δ_pp | among_skip cos≠full_m | among_skip cos≠full_ψ |"
                )
                lines.append(
                    "|-------------|---------:|--------:|-------:|-------:|---------:|--------:|-----:|----------------------:|----------------------:|"
                )
            for r in rows:
                if has_learned:
                    lines.append(
                        f"| {100*r['skip_rate_target']:.1f}% | {r['tau_margin']:.5f} | "
                        f"{r['delta_slack_local']:.5f} | {r['tau_learned_prob']:.5f} | "
                        f"{100*r['skip_rate_empirical_margin']:.2f} | "
                        f"{100*r['skip_rate_empirical_psi']:.2f} | "
                        f"{100*r['skip_rate_empirical_learned']:.2f} | "
                        f"{100*r['pct_disagree_with_full_margin_gate']:.2f} | "
                        f"{100*r['pct_disagree_with_full_psi_gate']:.2f} | "
                        f"{100*r['pct_disagree_with_full_learned_gate']:.2f} | "
                        f"{r['delta_disagree_psi_minus_margin_pp']:+.2f} | "
                        f"{r['delta_pp_learned_minus_psi']:+.2f} | "
                        f"{100*r['among_skipped_pct_cos_neq_full_margin']:.2f} | "
                        f"{100*r['among_skipped_pct_cos_neq_full_psi']:.2f} | "
                        f"{100*r['among_skipped_pct_cos_neq_full_learned']:.2f} |"
                    )
                else:
                    lines.append(
                        f"| {100*r['skip_rate_target']:.1f}% | {r['tau_margin']:.5f} | "
                        f"{r['delta_slack_local']:.5f} | {100*r['skip_rate_empirical_margin']:.2f} | "
                        f"{100*r['skip_rate_empirical_psi']:.2f} | "
                        f"{100*r['pct_disagree_with_full_margin_gate']:.2f} | "
                        f"{100*r['pct_disagree_with_full_psi_gate']:.2f} | "
                        f"{r['delta_disagree_psi_minus_margin_pp']:+.2f} | "
                        f"{100*r['among_skipped_pct_cos_neq_full_margin']:.2f} | "
                        f"{100*r['among_skipped_pct_cos_neq_full_psi']:.2f} |"
                    )
            lines.append("")
    lines.append(
        "> **Reading:** Lower **dis_ψ** at the same skip budget supports a better frontier "
        "than margin-only gating when disagreement with full rerank is the cost metric.\n"
    )
    lines.append(f"\n---\n*Generated: {datetime.now().isoformat()}*\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", type=str, default="coco_captions")
    ap.add_argument("--backbones", type=str, default="clip,siglip")
    ap.add_argument("--directions", type=str, default="i2t,t2i")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--cert-local-topk", type=int, default=50)
    ap.add_argument(
        "--skip-rates",
        type=str,
        default="0.15,0.25,0.35,0.45",
        help="Comma-separated target skip fractions in (0,1).",
    )
    ap.add_argument("--csls-k", type=int, default=20)
    ap.add_argument("--qb-tau", type=float, default=20.0)
    ap.add_argument("--db-tau", type=float, default=20.0)
    ap.add_argument("--nnn-k", type=int, default=64)
    ap.add_argument("--nnn-w", type=float, default=0.5)
    ap.add_argument(
        "--loader",
        type=str,
        choices=("standard", "laion"),
        default="standard",
        help="Embedding layout: standard NPZ rows vs LAION split files.",
    )
    ap.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Subset first N queries (0=all). Bounds LAION matrix size.",
    )
    ap.add_argument(
        "--max-gallery",
        type=int,
        default=0,
        help="Subset first M gallery items (0=all).",
    )
    ap.add_argument("--laion-query-split", type=str, default="test")
    ap.add_argument(
        "--laion-gallery-splits",
        type=str,
        default="test",
        help="Comma splits for LAION gallery shards (e.g. test,val,train).",
    )
    ap.add_argument(
        "--learned-logistic",
        action="store_true",
        help="Train sklearn logistic on train split; evaluate margin/ψ/learned on eval split.",
    )
    ap.add_argument("--learned-train-frac", type=float, default=0.5)
    ap.add_argument("--learned-seed", type=int, default=0)
    args = ap.parse_args()

    skip_rates = [float(x.strip()) for x in args.skip_rates.split(",") if x.strip()]
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    backbones = [x.strip() for x in args.backbones.split(",") if x.strip()]
    directions = [x.strip() for x in args.directions.split(",") if x.strip()]
    laion_gallery_splits = [
        x.strip() for x in args.laion_gallery_splits.split(",") if x.strip()
    ]

    cells = []
    for b in backbones:
        for d in directions:
            print(f"[cell] {b}|{d}", flush=True)
            cells.append(
                evaluate_cell(
                    args.dataset,
                    d,
                    b,
                    device,
                    args.cert_local_topk,
                    skip_rates,
                    args.csls_k,
                    args.qb_tau,
                    args.db_tau,
                    args.nnn_k,
                    args.nnn_w,
                    args.loader,
                    args.max_queries,
                    args.max_gallery,
                    args.laion_query_split,
                    laion_gallery_splits,
                    args.learned_logistic,
                    args.learned_train_frac,
                    args.learned_seed,
                )
            )

    cfg = {
        "dataset": args.dataset,
        "loader": args.loader,
        "cert_local_topk": args.cert_local_topk,
        "skip_rates": skip_rates,
        "csls_k": args.csls_k,
        "qb_tau": args.qb_tau,
        "db_tau": args.db_tau,
        "nnn_k": args.nnn_k,
        "nnn_w": args.nnn_w,
        "max_queries": args.max_queries,
        "max_gallery": args.max_gallery,
        "laion_query_split": args.laion_query_split,
        "laion_gallery_splits": laion_gallery_splits,
        "learned_logistic": args.learned_logistic,
        "learned_train_frac": args.learned_train_frac,
        "learned_seed": args.learned_seed,
        "generated": datetime.now().isoformat(),
    }
    payload = {"scope_note": SCOPE_NOTE, "config": cfg, "cells": cells}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"MarginVsPsiGate_{args.dataset}"
    js = OUTPUT_DIR / f"{stem}.json"
    md = OUTPUT_DIR / f"{stem}.md"
    js.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(md, args.dataset, cells, cfg)
    print(f"Wrote:\n  {js}\n  {md}")


if __name__ == "__main__":
    main()
