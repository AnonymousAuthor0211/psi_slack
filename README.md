# ψ-Slack: Selective Hubness Correction via Boundary-Localization Certificates

Anonymized code for the NeurIPS 2026 submission *"Reranking Acts at the Boundary: Selective Hubness Correction via ψ-Slack"*.

We study four training-free additive hubness rerankers (CSLS, QB-Norm, DB-Norm, NNN) for cross-modal retrieval and show they are **boundary-localized**: a top-1 change can occur only when the cosine margin is smaller than a per-query *ψ-slack* quantity. This yields a three-tier certificate hierarchy with sound, parameter-free no-op guarantees for R@1, R@K, MRR@K, and nDCG@K. On a 543K-gallery LAION benchmark the global certificate skips 25–36% of CSLS/NNN reranks losslessly, and the exact max-form certifies 68–91% of fixed top-50 pool outputs unchanged at exact pool fidelity (3.07–10.32× reranker-stage speedup over pool reranking).

## Observation (ψ-slack)

Although these methods are usually deployed as global score transformations, they are operationally **boundary-localized**: a top-1 change can happen only when the cosine margin is smaller than a per-query quantity we call **ψ-slack**:

$$B_q := \psi(c_1) - \min_c \psi(c).$$

We organize guarantees into a **three-tier certificate hierarchy**:

- **Theorem 1** — sound parameter-free no-op certificate for R@1 at full-gallery scope.
- **Proposition 2** — exact candidate-local max-form certificate for fixed-shortlist (e.g., top-50 pool) reranking.
- **Theorems 2–3** — sound top-K *set* and *order* invariance certificates lifting to R@K, MRR@K, nDCG@K.

This repository packages everything needed to **install once** and **rerun the analyses** below.

## Layout

| Path | Role |
|------|------|
| `psi_slack_pkg/` | Importable library: COCO embedding loaders, LAION NPZ loaders, `noop_certificate_speedup` (CSLS / QB / DB / NNN ψ and certificates). |
| `experiments/` | CLI scripts for ψ-slack experiments, LAION gating, plotting, and order-/set-invariance. |
| `scripts/laion/` | **LAION-2B-en → `laion_sample` pipeline** (metadata download, reservoir CSV, WebDataset→layout, dedupe, manifest). See `scripts/laion/README.md`. |
| `scripts/verify_laion_embeddings_metadata.py` | Sanity-check `embeddings_laion/laion_sample_*_{image,text}.npz` against `paper_draft/info.md` (model id, dim, ids). |
| `cache_embeddings_multi_gpu.py` | Multi-GPU embedding cache launcher (`embedders/`). Run from this repo root. |
| `evaluation_results/` | Default JSON/NPZ outputs (created when you run experiments). |
| `paper_draft/info.md` | Reproducibility supplement (checkpoints, LAION counts, manifest checklist). |
| `fetch_datasets.py`, `cache_embeddings.py`, `embedders/` | COCO / Flickr / audio helpers and shared embedders (OpenCLIP LAION CLIP, SigLIP, BLIP, CLAP, …). |

## Quick install

```bash
cd psi_slack   # repository root
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optional extras:

```bash
pip install -e ".[embed]"       # transformers + open_clip + timm (+ embedders stack)
pip install -e ".[laion_data]"  # HF datasets/hub + pandas/pyarrow for `scripts/laion/*.py`
pip install -e ".[xgboost]"     # stronger learned gate in `learned_gates_matched_skip` (sklearn fallback exists)
```

OpenAI CLIP (`embedders/clip_embedder.py`) needs the `clip` package (install from the [OpenAI CLIP](https://github.com/openai/CLIP) repository); it is not pinned in `pyproject.toml`.

External CLIs used only for LAION images: **`img2dataset`**, **`webdataset`** — install separately and record versions in your reproducibility bundle (`paper_draft/info.md`).

Conda users can start from `environment.yml` (CUDA PyTorch channel pins); rename or edit the env name locally.

## Data you must provide

All retrieval code assumes **pre-normalized** embeddings as NPZ files:

```
embeddings_<backbone>/
  <dataset>_test_image.npz   # keys: embeddings [N,D], ids
  <dataset>_test_text.npz
```

**COCO:** `dataset=coco_captions` → stems `coco_captions_test_{image,text}.npz` under each backbone folder.

**LAION sample:** `embeddings_<backbone>/laion_sample_{split}_{image,text}.npz` (see `psi_slack_pkg/laion.py`). IDs follow **`laion_sample_{split}_{i}`** on the image side and **`laion_sample_{split}_{i}_cap0`** on the text side (single caption per image), matching `cache_embeddings_multi_gpu.py`.

Use **`cache_embeddings_multi_gpu.py`** (recommended for LAION parity), **`cache_embeddings.py`** / **`fetch_datasets.py`** for smaller setups, or place compatible NPZs directly under `embeddings_*`.

### LAION subset: build `laion_sample/` + cache embeddings (aligned with `paper_draft/info.md`)

1. Follow **`scripts/laion/README.md`** end-to-end (metadata → `sample_laion2b_en_1m.py` → img2dataset → `webdataset_to_laion_sample.py` → optional dedupe → **`build_laion_sample_manifest.py`**).
2. Install backends: `pip install -e ".[embed,laion_data]"` plus **`open_clip_torch`** (included in `[embed]`).
3. Encode with the **same OpenCLIP hub string as the paper** (`hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K`, fp32, `preprocess_val`, autocast off — see `embedders/laion_embedder.py`):

```bash
python cache_embeddings_multi_gpu.py \
  --model_type laion \
  --datasets laion_sample \
  --data_root datasets/dataset_experiment \
  --output_dir embeddings_laion \
  --num_gpus 1 --gpu_ids 0 \
  --batch_size 128 --text_batch_size 512 --index_workers 32
```

4. Verify NPZs against the reproducibility doc:

```bash
python scripts/verify_laion_embeddings_metadata.py --embed-dir embeddings_laion --check-norms
```

Headline **gallery/query counts** for tables (**59 222 × 543 307**, etc.) are spelled out in **`paper_draft/info.md`** §1 and §6.

## Experiment scripts (what reproduces what)

Run from **repository root** (`psi_slack/`):

| Goal | Command |
|------|---------|
| **Per-query B_q / slack NPZs** (input to B_q figures) | `python experiments/cert_slack_histogram_experiment.py --dataset coco_captions --json-out evaluation_results/tables_GPU/CertSlackHistogram_coco.json --npz-dir evaluation_results/tables_GPU/cert_slack_per_query` |
| **B_q distribution + (m,B_q) scatter + collapse** | `python experiments/plot_bq_distribution.py --per-query-dir evaluation_results/tables_GPU/cert_slack_per_query --out paper_draft/figures/bq_distribution.pdf` |
| **Flip-colored (m,B_q)** | `python experiments/plot_bq_distribution.py --row-b-color-by flip --margin-n-bands 4 --anchor-cell clip,t2i --out paper_draft/figures/bq_distribution_flip_rowb.pdf` |
| **Margin vs ψ-slack @ matched skip** (Pareto-style disagreement table) | `python experiments/compare_margin_vs_psi_gate.py --dataset coco_captions --backbones clip --directions i2t,t2i --cert-local-topk 50` |
| **Learned gates vs ψ-slack** | `python experiments/learned_gates_matched_skip.py --dataset coco_captions --device 0` |
| **No-op certificate speedup + optional per-query margin dumps** | `python -m psi_slack_pkg.noop_certificate_speedup --dataset coco_captions --backbones clip,siglip` |
| **Margin bucket curves from JSON** | `python experiments/plot_noop_margin_buckets_from_json.py evaluation_results/tables_GPU/NoOpCertificate_coco_captions_margin_curves.json -o evaluation_results/figures_noop` |
| **Margin localization plots** (uses `--cert-per-query-dir` dumps) | `python experiments/margin_localization_report.py --npz-dir evaluation_results/tables_GPU/cert_per_query_coco --out-dir evaluation_results/tables_GPU/margin_localization` |
| **Slack histogram export only** | `python experiments/export_certificate_slack_histogram.py --dataset coco_captions --backbones clip --directions i2t,t2i` |
| **Order-invariance / set certificates (K sweep)** | `python experiments/order_invariance_certificate_k10.py --dataset coco_captions --K 2,5,10` |
| **LAION selective reranking + global / local scalar / Prop.2 max-form** | `python experiments/laion_gating_comparison.py --dataset laion_sample --backbone laion --direction t2i --gallery-splits test,val,train --cert-local-strict --output-tag selective` |

Legacy alias: `python experiments/compute_order_invariance.py` forwards to `order_invariance_certificate_k10`.

## LAION note

`experiments/laion_gating_comparison.py` uses streaming evaluation when `N_q × N_g × 4` bytes exceed `--max-sims-gb`; use a GPU with enough VRAM for ψ passes or reduce splits. **ANN / approximate nearest-neighbour matched-skip tooling** (FAISS / HNSW-style pipelines) is **not included** in this repository; this bundle covers subset construction, embedding caching, and ψ-slack / certificate analyses on **frozen NPZs**.

## Not bundled here

**SRP / matched abstention / ranking disagreement** stacks that pull large auxiliary dependencies are **not** copied here to keep the install footprint small.

## Citation

If you use this bundle academically, cite the paper and (once public) this repository.
