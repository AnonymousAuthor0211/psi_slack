# Reproducibility supplement (repository-derived)

This document gathers **frozen reproducibility choices**, **what remains inherently stochastic**, and **what reviewers asked for** (manifests, checkpoint IDs, splits). Paths are relative to **this repository root**: `experiments/`, `psi_slack_pkg/`, `scripts/`, `embedders/`, and `datasets/` unless noted.

---

## 1. LAION-543K / `laion_sample` headline subset

### What “543K” is in practice

Headline **`laion_sample`** JSON artifacts record **`n_queries=59 222`** and **`n_gallery=543 307`** when queries come from **`test`** and the gallery is the concatenation of **`test`, `val`, `train`** (e.g. `evaluation_results/tables_GPU/LaionANN_MatchedSkip_*_test_val_train.json`). A full-tree **`manifest.csv`** counts **every** split on disk (**`train`, `train_calib`, `val`, `test`**), so **`n_pairs`** exceeds **`n_gallery`** by roughly the **`train_calib`** size. **Quote archived JSON + manifest checksum** for the paper; the informal “543K” label is not a frozen HF row slice — it is the **union of embedding shards** built from **your local** `laion_sample/{test,val,train}` gallery tree (or equivalent `.npz` files).

### Pipeline (procedural)

Documented end-to-end in `scripts/laion/README.md`:

| Stage | Script / tool | Deterministic knobs |
|-------|----------------|---------------------|
| Metadata | `scripts/laion/download_laion2b_en_metadata.py` | HF **`--repo-id`** (default `laion/laion2B-en`), optional **`--revision`** (commit hash / branch). Filenames listed before download (`--list-only`). |
| Reservoir sample | `scripts/laion/sample_laion2b_en_1m.py` | **`--seed`** (default **42**), **`--sample-size`**, optional **`--english-only`**, column overrides **`--url-col` / `--caption-col`**. Uniform reservoir over **eligible** rows (non-empty URL + text). |
| Image fetch | `img2dataset` (external) | Example README settings: **`--image_size 256`**, **`--resize_mode no`**, **`--encode_format jpg`**, caption/url columns. **URL availability is time-dependent** — failed URLs never appear in the local corpus. |
| Layout → `laion_sample` | `scripts/laion/webdataset_to_laion_sample.py` | **`--seed`** (default **42**), **`--split-fracs`** default **`0.72,0.08,0.10,0.10`** for `train,train_calib,val,test`. Assigns each decoded `(jpg_bytes, caption)` pair to a split **i.i.d. per-sample** (not stratified by URL domain). **Numeric stems** `00000000.jpg` / `.txt` — **original LAION URLs are not retained as filenames**. |
| Corrupt filter | same script | Skips samples **PIL cannot decode** as RGB. |
| Optional dedup | `scripts/laion/dedupe_laion_sample_by_hash.py` | **SHA256 over raw JPEG bytes**; canonical split priority **train > train_calib > val > test**; removes duplicate copies across splits. |

### What the PDF should add for reproducibility

Reviewers are correct: the subset is **not** reconstructible from “543K” alone.

**Minimum artifacts to publish alongside the paper (recommended checklist):**

1. **`manifest.csv`** — generate with **`scripts/laion/build_laion_sample_manifest.py`**: one row per retained `(split, stem)` with **`split`**, **`stem`**, **`sha256_jpeg`**, **`caption_sha256`**, **`jpeg_bytes`**, **`caption_bytes`**, and optional **`laion_meta`** (merge via **`--stem-meta-csv`** if you preserved packed JSON from img2dataset / reservoir). Rows are emitted in fixed split order then lexicographic stem (deterministic “sorted manifest”).
2. **`MANIFEST.sha256`** — the same script writes a sidecar listing **`manifest_csv_sha256`** (hash of the exact UTF-8 bytes of `manifest.csv`), optional **`sample_laion2b_en_1m_csv_sha256`** / path (**`--sample-csv`**), and optional **`repo_git_tag`** (**`--repo-tag`** / **`--git-describe`**). Use **`manifest.parquet`** only if you convert from this canonical CSV for tooling convenience — treat the CSV hash as the archival anchor.
3. **Pinned metadata revision** — exact **`laion/laion2B-en`** git revision used + **`--max-shards`** if partial corpus + **`--sample-size`** + **`--seed`**.
4. **img2dataset version + CLI string** — capture full command-line (versions drift).
5. **`webdataset_to_laion_sample.py` invocation** — **`--seed`**, **`--split-fracs`**, **`--shard-glob`** at conversion time.
6. **Post-hoc**: whether **`dedupe_laion_sample_by_hash.py`** was run — if yes, ship **`DEDUPE_HASH_REPORT.txt`** or equivalent summary.

**Built-in sanity tooling (already in repo):**  
`scripts/laion/sanity_check_laion_sample.py` compares caption-length distributions vs original CSV, duplicate pixels across splits, and optional perceptual near-dup sampling (`imagehash`). Use **`--retained-metadata`** when you have img2dataset success metadata.

### Stable IDs used **inside** this codebase after caching

Embeddings are stored as `embeddings_laion/laion_sample_{split}_{image|text}.npz` with string **`ids`** such as **`laion_sample_{split}_{integer}`** on the image side and caption-side IDs carrying **`_cap{k}`** suffixes consistent with **`cache_embeddings_multi_gpu.py`** / `process_laion_split`. Evaluation scripts (**`experiments/laion_gating_comparison.py`**, etc.) pair GT via **`_cap` stripping** rules analogous to COCO.

---

## 2. Model checkpoints, preprocessing, precision

Below are **defaults wired into `cache_embeddings_multi_gpu.py`** (repository root) via **`embedders/`**. Each merged `.npz` stores an embedded **`metadata`** dict from `embedder.get_metadata()` where implemented — **archive those blobs** for audit.

### LAION CLIP (primary LAION backbone in caching scripts)

| Item | Source |
|------|--------|
| **Checkpoint id** | **`hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K`** (OpenCLIP hub string). |
| **Loader** | `embedders/laion_embedder.py` → `open_clip.create_model_and_transforms` + `open_clip.get_tokenizer(model_name)`. |
| **Image preprocessing** | **`preprocess_val`** (validation/inference transform), **RGB PIL**, **float32 stack**, autocast **disabled** on Laion path (`autocast(enabled=False)` — suitable for **V100 without bf16 flash**). |
| **Numeric precision** | Default **`dtype=torch.float32`** for LAION embedder (explicit for **V100** compatibility in caching script). |
| **Text** | OpenCLIP tokenizer from **`get_tokenizer(model_name)`** (model-specific truncation rules inside OpenCLIP). |
| **Output** | **L2-normalized** float vectors saved as **`np.float32`** in numpy (encoder outputs `.float().numpy()`). |

### OpenAI CLIP (`embeddings_clip` path)

| Item | Source |
|------|--------|
| **Default model name** | **`ViT-L/14@336px`** (`embedders/clip_embedder.py`). |
| **Loader** | `clip.load(model_name, device=..., download_root=...)`. |
| **Images** | Built-in **`self.preprocess`** from CLIP; batches fed as **`float32`** on GPU for ViT. |
| **Texts** | **`clip.tokenize(..., truncate=True)`**, **`context_length`** from loaded model. |
| **Encode dtype / storage** | Model keeps fp32 weights common path; stored embeddings often **`float16` numpy** in embedder path shown — **check each `.npz` dtype**. |

### OpenCLIP ViT-H/14 (`openclip_vit_h14`)

| Item | Source |
|------|--------|
| **Default** | **`hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K`** (`embedders/open_clip_embedder.py`). |

### EVA-CLIP L/14 (via OpenCLIP stack)

| Item | Source |
|------|--------|
| **Default** | **`hf-hub:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k`** (`cache_embeddings_multi_gpu.py` → **`eva_clip_l14`**). |

### SigLIP

| Item | Source |
|------|--------|
| **Default HF id** | **`google/siglip-base-patch16-256`** (`SiglipEmbedder`; overridden by **`--model_path`**). |
| **Processor** | **`SiglipProcessor.from_pretrained(..., use_fast=True)`** — resizing/padding/tokenization follow HF config (**patch16**, **`image_size`** from **`vision_config`**). |
| **Weights dtype** | **`torch_dtype=dtype`** passed into **`from_pretrained`** (default **float32** in constructor). |

### BLIP (retrieval head)

| Item | Source |
|------|--------|
| **Default path** | **`base_model/blip-itm-retrieval`** (local directory expected). |
| **Class** | **`BlipForImageTextRetrieval`** + **`BlipProcessor`** (**transformers**). |
| **Precision** | Default **`float16`** for encode path in **`BLIPEmbedder`**. |

### CLAP (Clotho / AudioCaps)

| Item | Source |
|------|--------|
| **Default path** | **`base_model/clap-htsat-fused`** (**float16** + **`max_duration=10.0`** in caching launcher). |

### Software pinning (what reviewers want)

This repository **does not** freeze one canonical lockfile for every optional pipeline. **For reproducibility**, capture at paper freeze:

- **`python --version`**
- **`pip freeze`** or **`conda env export`**
- **`torch`**, **`torchvision`**, **`open_clip_torch`**, **`clip-by-openai`**, **`transformers`**, **`datasets`**, **`img2dataset`**, **`faiss-cpu`/`faiss-gpu`**, **`hnswlib`** (optional ANN stack — pin alongside **`numpy`** major to avoid FAISS SWIG ABI mismatches).

---

## 3. COCO (Karpathy test) — split & retrieval protocol

### Corpus identity

The checked-in HF dataset snapshot lives under **`datasets/coco_captions/`** (Karpathy splits).  
`datasets/coco_captions/test/dataset_info.json` cites **`yerevann/coco-karpathy`** and records **`test`: 5000 examples** (images / grouped captions per Karpathy row).

### Embedding naming & splits

`psi_slack_pkg.embeddings.load_embeddings` loads:

- **`embeddings_{backbone}/coco_captions_test_image.npz`**
- **`embeddings_{backbone}/coco_captions_test_text.npz`**

So headline evaluations use **Karpathy `test` only** unless configs explicitly concatenate splits.

### Example / gallery cardinality

- **Images:** **5000** rows (`coco_captions_test_{i}` keys on image side).  
- **Texts:** **5000 × 5 = 25 000** captions (`coco_captions_test_{i}_cap{0..4}` convention from caching iterator).

### Ground truth & Recall / Top-1 semantics

`build_gt_mapping` (**`psi_slack_pkg.embeddings`**) and **`_correct`** (**`psi_slack_pkg/noop_certificate_speedup.py`**):

- **Image→text (`i2t`):** query id is **base image id**; gallery texts sharing that base id (all five captions) are **positives**.  
- **Text→image (`t2i`):** query id **`…_capk`** maps to **one** image id as positive.

Retrieval correctness uses **whether predicted gallery index is in the positive list** (standard multi-caption MSCOCO Karpathy).

---

## 4. Flickr30k (-Entities naming)

- Embedding NPZ stem defaults to **`flickr30k`** (`coco_captions`-style layout): **`flickr30k_test_image.npz`**, **`flickr30k_test_text.npz`**.  
- **`flickr30k_entities`** is accepted as a **dataset alias** mapping to the **`flickr30k`** stem when loading cached embeddings (see **`_dataset_embedding_npz_stem`**).  
- **Important:** Reproducibility requires stating whether experiments used **Karpathy Flickr30k splits** cached under `flickr30k_*` vs true **Flickr30k Entities** annotations — this codebase prioritizes **cached NPZ naming** above entity graphs unless separate pipelines say otherwise.

---

## 5. Clotho & AudioCaps

- Primarily via **`--model_type clap`** and **`embeddings_clap/`** (`clotho_*_{audio,text}.npz`).  
- Default checkpoint path above (**`base_model/clap-htsat-fused`**) + **10 s** clip duration cap in launcher.  
- Exact Clotho **subset name** (evaluation vs development) must match **`dataset/` layout** used when caching — **quote the split subdirectory & HF revision** used to build `datasets/` if publishing.

---

## 6. LAION evaluation protocols (how tables relate to code)

- **Gating / strategies:** **`experiments/laion_gating_comparison.py`** — streaming vs dense controlled by **`--max-sims-gb`**; gallery concatenation **`--gallery-splits test,val,train`**.  
- **ANN matched skip:** **Not included** in this repository (optional extension using FAISS IVF-PQ / IVF-Flat or **hnswlib** HNSW; protocol described in the paper — pin **`--nprobe`** / **`ef`**, normalisation, and NumPy major when reproducing).  
- **Reproducibility bundle:** after freezing **`laion_sample`**, run **`scripts/laion/build_laion_sample_manifest.py`** (see §1 checklist) so table **`n_gallery`** matches **`manifest.csv`**.  
- **Headline counts:** prior JSON artifacts reported **`n_queries=59222`, `n_gallery=543307`** for `laion_sample` combined gallery — **tie numbers to your shipped manifest**, not hand-wavy “543K”.

---

## 7. Suggested PDF edits (short checklist)

1. Replace “543K subset” with **exact gallery count** from **`n_gallery`** in archived JSON + **manifest checksum**.  
2. Add **Table: Backbone → HF hub string / OpenAI name → embedding dtype → image px**.  
3. State **COCO Karpathy test**, **5000 / 25 000**, **multi-caption positives**.  
4. State **Flickr30k** cache stem **`flickr30k`** vs Entities naming ambiguity.  
5. Pin **`laion2B-en`** revision + **`sample_laion2b_en_1m --seed`** + **`webdataset_to_laion_sample --seed`** + **`split-fracs`** + dedupe yes/no.  
6. Supply **`pip freeze`** or **`conda export`** for the submission bundle.

---

*Generated from repository inspection; update checksums and shard revisions when you freeze the camera-ready artifact.*
