# LAION-2B-en → `laion_sample` pipeline

Run all commands from **this repository root** (`psi_slack/`). Embedding caching uses **`cache_embeddings_multi_gpu.py`** at the repo root; data-prep scripts live under **`scripts/laion/`**. After caching, run **`python scripts/verify_laion_embeddings_metadata.py`** and read **`paper_draft/info.md`**.

This folder wires **parquet metadata → CSV → img2dataset (WebDataset) → `laion_sample/{train,...}/*.jpg` + `.txt`**, matching what **`cache_embeddings_multi_gpu.py`** expects.

## Dependencies

```bash
pip install huggingface_hub tqdm pandas pyarrow img2dataset webdataset pillow
```

## 1) Download metadata shards (parquet only)

```bash
python scripts/laion/download_laion2b_en_metadata.py --list-only
python scripts/laion/download_laion2b_en_metadata.py \
  --output-dir data/laion2b_en_meta --max-shards 8   # smoke test
# Full corpus: omit --max-shards (huge disk + HF terms)
```

## 2) Reservoir sample 1M eligible rows → CSV

Uniform sampling over **eligible** rows (non-empty URL + caption; auto-detects `URL`/`TEXT` or `url`/`caption`/`text`; optional `--english-only`).

```bash
python scripts/laion/sample_laion2b_en_1m.py \
  --meta-dir data/laion2b_en_meta \
  --out-csv data/laion2b_en_1m_sample.csv \
  --sample-size 1000000 \
  --seed 42
```

Debug on a few shards:

```bash
python scripts/laion/sample_laion2b_en_1m.py \
  --meta-dir data/laion2b_en_meta \
  --out-csv data/small.csv \
  --sample-size 5000 \
  --max-eligible-rows 200000 \
  --seed 0
```

Optional provenance column for img2dataset (default `laion_meta` JSON): already included when metadata columns exist.

## 3) Download images with img2dataset

```bash
mkdir -p data/laion2b_en_1m_webdataset

img2dataset \
  --url_list data/laion2b_en_1m_sample.csv \
  --input_format csv \
  --url_col url \
  --caption_col caption \
  --output_folder data/laion2b_en_1m_webdataset \
  --output_format webdataset \
  --processes_count 16 \
  --thread_count 64 \
  --image_size 256 \
  --resize_mode no \
  --encode_format jpg \
  --save_additional_columns '["laion_meta"]' \
  --enable_wandb False
```

If you did **not** add `laion_meta` to the CSV, omit `--save_additional_columns` or pass only columns that exist.

## 4) Convert WebDataset → dataset layout

Writes `datasets/dataset_experiment/laion_sample/{train,...}/NNNNNNNN.jpg` + `.txt`.

```bash
python scripts/laion/webdataset_to_laion_sample.py \
  --shard-glob 'data/laion2b_en_1m_webdataset/*.tar' \
  --data-root datasets/dataset_experiment \
  --seed 42 \
  --split-fracs 0.72,0.08,0.10,0.10
```

Skips samples that **PIL cannot decode** (minimal corrupt-image filter).

## 5) Cache embeddings

```bash
python cache_embeddings_multi_gpu.py \
  --model_type laion \
  --datasets laion_sample \
  --data_root datasets/dataset_experiment \
  --output_dir embeddings_laion \
  --num_gpus 1 \
  --gpu_ids 0
```

**Multi-GPU / pod stability**

- **`--gpu_ids` must match real device indices** on the machine (e.g. four GPUs are usually `0,1,2,3`, not `1,2,3,4` unless GPU0 is reserved). You must pass **at least** `--num_gpus` ids (e.g. six GPUs: `--num_gpus 6 --gpu_ids 0,1,2,3,4,5`).
- **Default schedule (`--model_type laion`):** `sequential_sharded` — processes **one split at a time**, uses **all GPUs** to shard that split by row range, merges into `laion_sample_{split}_image.npz` / `_text.npz`, then moves to the next split. Order is **smallest split first**, **`train` last** so you see finished `.npz` files quickly before the long train encode.
- **Legacy:** `--laion_schedule round_robin` assigns whole splits to different GPUs (old behavior; can leave GPUs idle when splits differ in size).
- **Saves:** each split ends as merged `laion_sample_{split}_image.npz` and `laion_sample_{split}_text.npz` after that split’s workers finish. There is no per-batch `.npz` checkpoint; crashes mid-split lose that split’s outputs (re-run).
- **RAM:** LAION caching passes **file paths** into the embedder (batched PIL loads). Avoids loading every image into CPU RAM before encode (which OOMs large splits + multiple workers).

**Speed (you are already using GPUs for encode)**

- Image and text **forward passes run on the GPU** (`encode_images` / `encode_texts`). What feels “slow” is often (1) **indexing** hundreds of thousands of `.txt` files (now parallelized via `--index_workers`), (2) **disk read** of JPEGs, (3) **`train`** being ~10× larger than other splits — one GPU works that split for a long time while others may finish early.
- Try: `--batch_size 256` (if VRAM allows), `--text_batch_size 384` or `512` (text tower is often lighter than the image batch).
- **Six GPUs (sequential sharding, default):**

```bash
python cache_embeddings_multi_gpu.py \
  --model_type laion --datasets laion_sample \
  --data_root datasets/dataset_experiment --output_dir embeddings_laion \
  --num_gpus 6 --gpu_ids 0,1,2,3,4,5 \
  --batch_size 256 --text_batch_size 512 --index_workers 48
```

- Four-GPU example:

```bash
python cache_embeddings_multi_gpu.py \
  --model_type laion --datasets laion_sample \
  --data_root datasets/dataset_experiment --output_dir embeddings_laion \
  --num_gpus 4 --gpu_ids 0,1,2,3 \
  --batch_size 256 --text_batch_size 512 --index_workers 48
```

## 6) ψ-slack evaluation (this repository)

Run LAION gating / certificates from frozen NPZs:

```bash
python experiments/laion_gating_comparison.py --dataset laion_sample --backbone laion --direction i2t --gallery-splits test,val,train
```

## 7) Sanity check (original CSV vs retained `laion_sample`)

Compares distributions (caption length, optional similarity / width / height / domain / language), checks split disjointness, **exact** duplicate bytes across splits, test split size, and optional **perceptual** near-duplicate sampling (`imagehash`).

**Report file:** by default the same log is written to `<laion-root>/SANITY_CHECK_REPORT.txt` (and still printed in the terminal). Override with `--output-report PATH`, or use `--no-save-report` for stdout only.

```bash
pip install pandas pyarrow pillow tqdm scipy  # scipy optional (KS test)

python scripts/laion/sanity_check_laion_sample.py \
  --original-csv data/laion2b_en_1m_sample.csv \
  --laion-root datasets/dataset_experiment/laion_sample \
  --retained-metadata path/to/img2dataset_success_subset.parquet

# Optional: subsample PIL reads; optional near-dup probe (not exhaustive)
python scripts/laion/sanity_check_laion_sample.py \
  --original-csv data/laion2b_en_1m_sample.csv \
  --laion-root datasets/dataset_experiment/laion_sample \
  --max-read-images 50000 \
  --near-dup-phash-sample 800 \
  --near-dup-hamming 6
```

Pass a **retained-metadata** parquet/CSV that matches the ~606K successful rows (same columns as img2dataset output: `url`, `caption`, `similarity`, `width`, `height`, …) to compare similarity, resolution, and URL domains to the original 1M. Without it, the script still compares **caption length** (original CSV vs on-disk `.txt`) and runs duplicate / split checks.

## 8) Deduplicate exact bytes across splits (one hash → one split)

If `SANITY_CHECK_REPORT.txt` shows duplicate SHA256 groups spanning train/val/test, run:

```bash
# Preview (writes DEDUPE_HASH_REPORT.txt, does not delete)
python scripts/laion/dedupe_laion_sample_by_hash.py \
  --laion-root datasets/dataset_experiment/laion_sample --dry-run

# Apply: canonical split = train if any copy in train, else train_calib, else val, else test
python scripts/laion/dedupe_laion_sample_by_hash.py \
  --laion-root datasets/dataset_experiment/laion_sample
```

Then **re-build `embeddings_laion`** (`cache_embeddings_multi_gpu.py`) so counts match the cleaned tree.

## 9) Manifest + `MANIFEST.sha256` (camera-ready archive)

After dedupe (if any), emit a sorted **`manifest.csv`** and checksum sidecar for reviewers:

```bash
python scripts/laion/build_laion_sample_manifest.py \
  --laion-root datasets/dataset_experiment/laion_sample \
  --out-dir datasets/dataset_experiment/laion_sample \
  --sample-csv data/laion2b_en_1m_sample.csv \
  --git-describe
```

- **`--sample-csv`**: records **`sample_laion2b_en_1m_csv_sha256`** in **`MANIFEST.sha256`** (same file produced by `sample_laion2b_en_1m.py` if you kept it).
- **`--stem-meta-csv`**: optional CSV with columns **`split,stem,laion_meta`** to attach img2dataset-style JSON per row (otherwise **`laion_meta`** is empty).

Archive **`manifest.csv`**, **`MANIFEST.sha256`**, and the reservoir CSV when publishing.

---

**Note:** Verify the Hugging Face dataset id and shard names for `laion2B-en` on the dataset card; use `--repo-id` / `--revision` on the downloader if the canonical repo moves.
