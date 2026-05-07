#!/usr/bin/env python3
"""
Experiment 1 figure: B_q distribution, certificate geometry, and cross-method collapse.

Reads per-query NPZs produced by ``experiments/cert_slack_histogram_experiment.py`` (default folder
``evaluation_results/tables_GPU/cert_slack_per_query``), filenames::

    <backbone>__<direction>__<MethodStem>__slack_<mode>.npz

with ``mode`` = ``global`` or ``localK50``. Arrays used: ``margin`` (= :math:`m`),
``B_q``, ``changed_full`` (flip iff cosine top-1 :math:`\\neq` full rerank top-1),
optionally ``certified`` / ``slack``.

Produces **bq_distribution.pdf** with three rows:

  **Row A** — Per-backbone CDFs of :math:`B_q` (pooling i2t + t2i), one curve per method.
  Shows concentration at small :math:`B_q` and tighter CSLS/NNN vs QB/DB.

  **Row B** — :math:`(m, B_q)` scatter on ``--anchor-cell`` (default ``clip,t2i``), one panel
  per method. The half-plane **certificate-firing** region :math:`m \\geq B_q` (below the
  diagonal in :math:`(m,B_q)` coordinates) is lightly shaded; diagonal :math:`m=B_q` drawn.

  Use ``--row-b-color-by flip`` to color points by **flip vs no-flip** (NPZ ``changed_full``:
  cosine top-1 :math:`\\neq` full-gallery rerank top-1), instead of certified vs not. Optional
  ``--margin-n-bands N`` draws faint vertical quantile lines on :math:`m(q)` splitting into
  :math:`N` margin bands (same firing-region shading).

  **Row C** — On ``--collapse-bb`` (default ``clip``), overlaid CDFs of rescaled slack
  :math:`m/\\mathrm{med}(m) - B_q/\\mathrm{med}(B_q)` (pooling both directions, per method).
  Universal boundary localization ⇒ overlapping curves.

CLI::

  python experiments/plot_bq_distribution.py \\
      --per-query-dir evaluation_results/tables_GPU/cert_slack_per_query \\
      --out paper_draft/figures/bq_distribution.pdf

  python experiments/plot_bq_distribution.py --mode local50 \\
      --out paper_draft/figures/bq_distribution_local50.pdf

  # (m, B_q) scatter colored by flip/no-flip + margin quantile bands:
  python experiments/plot_bq_distribution.py \\
      --row-b-color-by flip --margin-n-bands 4 \\
      --out paper_draft/figures/bq_distribution_flip_rowb.pdf

  # Compact figure: row B only (for a separate figure label):
  python experiments/plot_bq_distribution.py --row-b-color-by flip --margin-n-bands 4 \\
      --mbq-flip-scatter-only-out paper_draft/figures/m_bq_scatter_flip.pdf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

METHODS = ("CSLS", "QB-Norm", "DB-Norm", "NNN")
METHOD_STEMS = {
    "CSLS": "CSLS",
    "QB-Norm": "QB_Norm",
    "DB-Norm": "DB_Norm",
    "NNN": "NNN",
}
STEM_TO_METHOD = {v: k for k, v in METHOD_STEMS.items()}

METHOD_COLOR = {
    "CSLS": "#4C72B0",
    "QB-Norm": "#DD8452",
    "DB-Norm": "#CC8963",
    "NNN": "#55A868",
}

DEFAULT_BACKBONE_ORDER = ("clip", "siglip", "blip", "eva_clip_l14")

# NPZ mode segment as written by cert_slack_histogram_experiment.py
MODE_FILE_ALIASES = {
    "global": "global",
    "local50": "localK50",
    "localK50": "localK50",
}

# Must allow e.g. slack_localK50 (mixed case)
NPZ_RE = re.compile(
    r"^(?P<bb>[a-z0-9_]+)__(?P<dir>[a-z0-9]+)__(?P<stem>[A-Za-z_]+)__slack_(?P<mode>.+)\.npz$"
)


def _resolve_mode_file_tag(mode_arg: str) -> tuple[str, str]:
    """Returns (filename_mode_segment, LaTeX superscript label for plots)."""
    key = mode_arg.strip().lower().replace("-", "")
    if key in ("local50", "localk50"):
        return MODE_FILE_ALIASES["local50"], r"\mathrm{local\text{-}K50}"
    if key == "global":
        return "global", r"\mathrm{global}"
    raise ValueError(f"Unknown --mode {mode_arg!r}; use global or local50")


def _parse_anchor_cell(s: str) -> tuple[str, str]:
    """
    Row B backbone and direction. Use ``clip,t2i`` (comma is shell-safe) or
    ``clip|t2i`` (in bash you must quote: ``'clip|t2i'`` — otherwise ``|`` starts a pipe).
    """
    t = s.strip()
    if "|" in t:
        bb, d = t.split("|", 1)
    elif "," in t:
        bb, d = t.split(",", 1)
    else:
        raise SystemExit(
            f"--anchor-cell {s!r} needs two parts: backbone,direction or backbone|direction "
            f"(e.g. clip,t2i)."
        )
    bb, d = bb.strip(), d.strip()
    if not bb or not d:
        raise SystemExit(f"Invalid --anchor-cell {s!r}: empty backbone or direction.")
    return bb, d


def _index_per_query(per_query_dir: Path) -> dict[tuple[str, str, str, str], Path]:
    idx: dict[tuple[str, str, str, str], Path] = {}
    for f in per_query_dir.glob("*__slack_*.npz"):
        m = NPZ_RE.match(f.name)
        if not m:
            continue
        bb, direction, stem, mode = m["bb"], m["dir"], m["stem"], m["mode"]
        method = STEM_TO_METHOD.get(stem)
        if method is None:
            continue
        idx[(bb, direction, method, mode)] = f
    return idx


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.sort(values.astype(np.float64))
    n = len(v)
    y = np.arange(1, n + 1) / max(n, 1)
    return v, y


def _panel_A_cdfs(
    ax: plt.Axes,
    idx: dict,
    backbone: str,
    mode_file: str,
    bq_sup_label: str,
    title: str,
    show_legend: bool,
) -> None:
    for method in METHODS:
        chunks: list[np.ndarray] = []
        for direction in ("i2t", "t2i"):
            key = (backbone, direction, method, mode_file)
            if key not in idx:
                continue
            d = np.load(idx[key], allow_pickle=True)
            chunks.append(np.asarray(d["B_q"], dtype=np.float64).ravel())
        if not chunks:
            continue
        bq = np.concatenate(chunks)
        bq = bq[np.isfinite(bq)]
        if bq.size == 0:
            continue
        x, y = _ecdf(bq)
        ax.plot(x, y, color=METHOD_COLOR[method], label=method, lw=1.4)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"$B_q^{" + bq_sup_label + r"}$")
    ax.set_ylabel("CDF")
    ax.grid(alpha=0.25)
    if show_legend:
        ax.legend(fontsize=7, loc="lower right")


FLIP_COLOR = "#C44E52"
NO_FLIP_COLOR = "#4C72B0"
CERT_YES_COLOR = "#2CA02C"
CERT_NO_COLOR = "#C44E52"


def _panel_B_scatter(
    ax: plt.Axes,
    idx: dict,
    backbone: str,
    direction: str,
    method: str,
    mode_file: str,
    bq_sup_label: str,
    *,
    color_by: str = "cert",
    margin_n_bands: int = 0,
    rng_seed: int = 0,
    max_points: int = 12000,
) -> None:
    key = (backbone, direction, method, mode_file)
    if key not in idx:
        ax.text(0.5, 0.5, "missing NPZ", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    d = np.load(idx[key], allow_pickle=True)
    m = np.asarray(d["margin"], dtype=np.float64).ravel()
    bq = np.asarray(d["B_q"], dtype=np.float64).ravel()
    assert m.shape == bq.shape

    flip = None
    if color_by == "flip":
        if "changed_full" not in d.files:
            ax.text(
                0.5,
                0.5,
                "NPZ missing changed_full\n(run cert_slack_histogram_experiment.py)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.set_axis_off()
            return
        flip = np.asarray(d["changed_full"], dtype=bool).ravel()
        assert flip.shape == m.shape

    valid = np.isfinite(m) & np.isfinite(bq)
    m = m[valid]
    bq = bq[valid]
    if flip is not None:
        flip = flip[valid]

    if m.size == 0:
        ax.text(0.5, 0.5, "no finite points", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    if len(m) > max_points:
        sel = np.random.RandomState(rng_seed).choice(len(m), max_points, replace=False)
        m, bq = m[sel], bq[sel]
        if flip is not None:
            flip = flip[sel]

    lo = float(min(m.min(), bq.min()))
    hi = float(max(m.max(), bq.max()))
    pad = 0.03 * (hi - lo + 1e-12)
    x_lo, x_hi = lo - pad, hi + pad
    y_lo, y_hi = lo - pad, hi + pad
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # Certificate region m >= B_q  ⇔  points below/on diagonal y=x in (m,B_q) axes (x=m, y=B_q).
    xs = np.linspace(x_lo, x_hi, 400)
    y_top = np.maximum(y_lo, np.minimum(xs, y_hi))
    ax.fill_between(
        xs,
        y_lo,
        y_top,
        alpha=0.18,
        color="#BBBBBB",
        zorder=1,
        linewidth=0,
        label=r"$m \geq B_q$ (cert. fires)",
    )

    if margin_n_bands >= 2:
        qs = np.linspace(0.0, 1.0, margin_n_bands + 1)[1:-1]
        for q in qs:
            xv = float(np.quantile(m, q))
            ax.axvline(xv, color="0.45", lw=0.7, ls=":", alpha=0.55, zorder=2)

    ax.plot([x_lo, x_hi], [x_lo, x_hi], "k--", lw=0.9, alpha=0.65, zorder=2, label=r"$m=B_q$")

    if color_by == "cert":
        cert = m >= bq
        ax.scatter(
            m[~cert],
            bq[~cert],
            s=2,
            alpha=0.35,
            c=CERT_NO_COLOR,
            zorder=3,
            label="not certified",
        )
        ax.scatter(
            m[cert],
            bq[cert],
            s=2,
            alpha=0.35,
            c=CERT_YES_COLOR,
            zorder=3,
            label="certified",
        )
    else:
        assert flip is not None
        nf = ~flip
        ax.scatter(
            m[nf],
            bq[nf],
            s=2,
            alpha=0.4,
            c=NO_FLIP_COLOR,
            zorder=3,
            label=r"no flip ($i_{\mathrm{cos}}=i_{\mathrm{full}}$)",
        )
        ax.scatter(
            m[flip],
            bq[flip],
            s=3,
            alpha=0.65,
            c=FLIP_COLOR,
            zorder=4,
            label=r"flip ($i_{\mathrm{cos}}\neq i_{\mathrm{full}}$)",
        )
        # subtitle flip rate
        fr = float(np.mean(flip))
        ax.text(
            0.98,
            0.02,
            f"flip rate={100.0 * fr:.2f}%",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="0.2",
        )

    ax.set_title(f"{method} ({backbone}|{direction})", fontsize=9)
    ax.set_xlabel(r"$m(q)$")
    ax.set_ylabel(r"$B_q^{" + bq_sup_label + r"}(q)$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=5.5, loc="upper left")


def _write_mbq_flip_scatter_only_figure(
    idx_mode: dict,
    anchor_bb: str,
    anchor_dir: str,
    mode_file: str,
    bq_sup_label: str,
    margin_n_bands: int,
    out: Path,
    dpi: int,
) -> None:
    """Single row of (m, B_q) panels — flip-colored; for a standalone paper figure."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), constrained_layout=True)
    assert len(METHODS) == len(axes)
    for ax, method in zip(axes, METHODS):
        _panel_B_scatter(
            ax,
            idx_mode,
            anchor_bb,
            anchor_dir,
            method,
            mode_file,
            bq_sup_label,
            color_by="flip",
            margin_n_bands=margin_n_bands,
        )
    mode_title = (
        "global min $\\psi$" if mode_file == "global" else "candidate-local min $\\psi$ (top-50)"
    )
    fig.suptitle(
        rf"$(m, B_q)$ — colored by flip ($i_{{\mathrm{{cos}}}}\neq i_{{\mathrm{{full}}}}$) | "
        rf"$B_q^{{{bq_sup_label}}}$ ({mode_title})",
        fontsize=11,
        y=1.02,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _panel_C_collapse(
    ax: plt.Axes,
    idx: dict,
    backbone: str,
    mode_file: str,
    bq_sup_label: str,
) -> None:
    """CDF of m/med(m) - B_q/med(B_q), pooled over both directions per method."""
    for method in METHODS:
        chunks: list[np.ndarray] = []
        for direction in ("i2t", "t2i"):
            key = (backbone, direction, method, mode_file)
            if key not in idx:
                continue
            d = np.load(idx[key], allow_pickle=True)
            m = np.asarray(d["margin"], dtype=np.float64).ravel()
            bq = np.asarray(d["B_q"], dtype=np.float64).ravel()
            mm = float(np.median(m))
            mb = float(np.median(bq))
            if mm <= 0 or mb <= 0:
                continue
            slack_rescaled = (m / mm) - (bq / mb)
            chunks.append(slack_rescaled)
        if not chunks:
            continue
        slack = np.concatenate(chunks)
        slack = slack[np.isfinite(slack)]
        if slack.size == 0:
            continue
        x, y = _ecdf(slack)
        ax.plot(x, y, color=METHOD_COLOR[method], label=method, lw=1.4)

    ax.axvline(0.0, color="k", lw=0.9, ls="--", alpha=0.55)
    ax.set_title(
        rf"Cross-method collapse ({backbone}, $B_q^{{{bq_sup_label}}}$)",
        fontsize=10,
    )
    ax.set_xlabel(r"$m/\mathrm{med}(m) - B_q/\mathrm{med}(B_q)$")
    ax.set_ylabel("CDF")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--per-query-dir",
        type=Path,
        default=project_root / "evaluation_results/tables_GPU/cert_slack_per_query",
        help="Folder with cert_slack_histogram_experiment NPZs.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=project_root / "paper_draft/figures/bq_distribution.pdf",
        help="Output PDF path.",
    )
    ap.add_argument(
        "--anchor-cell",
        type=str,
        default="clip,t2i",
        help="Row B cell: 'backbone,direction' or 'backbone|direction' (comma avoids bash pipe).",
    )
    ap.add_argument("--collapse-bb", type=str, default="clip")
    ap.add_argument(
        "--mode",
        type=str,
        default="global",
        choices=("global", "local50"),
        help="global → slack_global.npz; local50 → slack_localK50.npz",
    )
    ap.add_argument(
        "--backbone-order",
        type=str,
        default=",".join(DEFAULT_BACKBONE_ORDER),
        help="Comma-separated backbone ids for row A columns (left to right).",
    )
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument(
        "--row-b-color-by",
        type=str,
        default="cert",
        choices=("cert", "flip"),
        help="Row B: color by certified vs not (default), or flip vs no-flip (NPZ changed_full).",
    )
    ap.add_argument(
        "--margin-n-bands",
        type=int,
        default=0,
        help="If >=2, draw vertical quantile lines on m(q) into N margin bands (e.g. 4 → quartiles).",
    )
    ap.add_argument(
        "--mbq-flip-scatter-only-out",
        type=Path,
        default=None,
        help="Optional extra PDF: one row of (m,B_q) scatters, always flip-colored.",
    )
    args = ap.parse_args()

    if args.margin_n_bands < 0:
        raise SystemExit("--margin-n-bands must be >= 0")
    margin_bands_use = args.margin_n_bands if args.margin_n_bands >= 2 else 0

    mode_file, bq_sup_label = _resolve_mode_file_tag(args.mode)

    per_query_dir = args.per_query_dir
    if not per_query_dir.is_absolute():
        per_query_dir = project_root / per_query_dir

    idx = _index_per_query(per_query_dir)
    idx_mode = {k: v for k, v in idx.items() if k[3] == mode_file}
    if not idx_mode:
        loose = sorted(per_query_dir.glob("*.npz"))
        msg = (
            f"No NPZs for mode '{mode_file}' under {per_query_dir}\n"
            f"Expected names like clip__t2i__CSLS__slack_{mode_file}.npz "
            f"(from experiments/cert_slack_histogram_experiment.py)."
        )
        if loose:
            msg += f"\nFound {len(loose)} *.npz (showing first 5): {[p.name for p in loose[:5]]}"
        raise SystemExit(msg)

    requested_bb = [x.strip() for x in args.backbone_order.split(",") if x.strip()]
    present_bb = {k[0] for k in idx_mode}
    row_a_backbones = [b for b in requested_bb if b in present_bb]
    for b in sorted(present_bb):
        if b not in row_a_backbones:
            row_a_backbones.append(b)

    n_col = min(4, max(len(row_a_backbones), 1))
    row_a_cols = row_a_backbones[:n_col]

    print(
        f"Mode file tag: {mode_file} | Row A backbones: {row_a_cols} | "
        f"indexed entries (this mode): {len(idx_mode)}"
    )

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)

    # ----- Row A -----
    for j in range(4):
        ax = fig.add_subplot(gs[0, j])
        if j < len(row_a_cols):
            bb = row_a_cols[j]
            _panel_A_cdfs(
                ax,
                idx_mode,
                bb,
                mode_file,
                bq_sup_label,
                title=f"{bb}: $B_q^{{{bq_sup_label}}}$",
                show_legend=(j == 0),
            )
        else:
            ax.set_visible(False)

    # ----- Row B -----
    anchor_bb, anchor_dir = _parse_anchor_cell(args.anchor_cell)
    for j, method in enumerate(METHODS):
        ax = fig.add_subplot(gs[1, j])
        _panel_B_scatter(
            ax,
            idx_mode,
            anchor_bb,
            anchor_dir,
            method,
            mode_file,
            bq_sup_label,
            color_by=args.row_b_color_by,
            margin_n_bands=margin_bands_use,
        )

    # ----- Row C -----
    ax_c = fig.add_subplot(gs[2, :])
    _panel_C_collapse(ax_c, idx_mode, args.collapse_bb.strip(), mode_file, bq_sup_label)

    mode_title = "global min $\\psi$" if mode_file == "global" else "candidate-local min $\\psi$ (top-50)"
    row_b_note = (
        " — Row B: flip/no-flip"
        if args.row_b_color_by == "flip"
        else ""
    )
    fig.suptitle(
        rf"$B_q$ distribution and boundary localization ({mode_title}){row_b_note}",
        fontsize=12,
        y=1.02,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.out}")

    if args.mbq_flip_scatter_only_out is not None:
        po = args.mbq_flip_scatter_only_out
        if not po.is_absolute():
            po = project_root / po
        _write_mbq_flip_scatter_only_figure(
            idx_mode,
            anchor_bb,
            anchor_dir,
            mode_file,
            bq_sup_label,
            margin_bands_use,
            po,
            args.dpi,
        )


if __name__ == "__main__":
    main()
