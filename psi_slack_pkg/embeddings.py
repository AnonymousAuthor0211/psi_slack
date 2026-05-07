"""
Minimal embedding loaders for Karpathy-style cross-modal retrieval benchmarks.

Expects precomputed NPZs under ``REPO_ROOT/embeddings_<backbone>/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from psi_slack_pkg._paths import REPO_ROOT as project_root


def _dataset_embedding_npz_stem(dataset: str) -> str:
    aliases = {
        "flickr30k_entities": "flickr30k",
        "flickr_entities": "flickr30k",
    }
    return aliases.get(dataset, dataset)


def load_embeddings(
    dataset: str, direction: str, backbone: str = "clip"
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Load query and gallery embeddings for a dataset/direction (test split)."""
    embed_dir = project_root / f"embeddings_{backbone}"
    stem = _dataset_embedding_npz_stem(dataset)

    if direction == "i2t":
        query_mod, gallery_mod = "image", "text"
    elif direction == "t2i":
        query_mod, gallery_mod = "text", "image"
    elif direction == "a2t":
        query_mod, gallery_mod = "audio", "text"
    elif direction == "t2a":
        query_mod, gallery_mod = "text", "audio"
    else:
        raise ValueError(f"Unknown direction: {direction}")

    query_file = embed_dir / f"{stem}_test_{query_mod}.npz"
    gallery_file = embed_dir / f"{stem}_test_{gallery_mod}.npz"

    if not query_file.exists() or not gallery_file.exists():
        hint = ""
        if dataset != stem:
            hint = f" (resolved stem `{stem}` from `{dataset}`)"
        raise FileNotFoundError(
            f"Embeddings not found{hint}:\n  {query_file}\n  {gallery_file}"
        )

    query_data = np.load(query_file)
    gallery_data = np.load(gallery_file)

    query_emb = torch.tensor(query_data["embeddings"], dtype=torch.float32)
    gallery_emb = torch.tensor(gallery_data["embeddings"], dtype=torch.float32)
    query_ids = query_data["ids"]
    gallery_ids = gallery_data["ids"]

    return query_emb, gallery_emb, query_ids, gallery_ids


def build_gt_mapping(
    query_ids: np.ndarray, gallery_ids: np.ndarray, direction: str
) -> Dict[str, List[int]]:
    """Build ground truth mapping from query IDs to gallery indices (COCO-style id conventions)."""
    gt_mapping: Dict[str, List[int]] = {}
    gallery_ids_list = [str(gid) for gid in gallery_ids]

    if direction in ["i2t", "a2t"]:
        base_to_gallery: Dict[str, List[int]] = {}
        for j, gid in enumerate(gallery_ids_list):
            if "_cap" in gid:
                base_id = gid.rsplit("_cap", 1)[0]
            else:
                base_id = gid
            base_to_gallery.setdefault(base_id, []).append(j)

        for qid in query_ids:
            qid_str = str(qid)
            if qid_str in base_to_gallery:
                gt_mapping[qid_str] = base_to_gallery[qid_str]
    else:
        gallery_id_to_idx = {gid: j for j, gid in enumerate(gallery_ids_list)}

        for qid in query_ids:
            qid_str = str(qid)
            if "_cap" in qid_str:
                base_id = qid_str.rsplit("_cap", 1)[0]
            else:
                base_id = qid_str

            if base_id in gallery_id_to_idx:
                gt_mapping[qid_str] = [gallery_id_to_idx[base_id]]

    return gt_mapping
