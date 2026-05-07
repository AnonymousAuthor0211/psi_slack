"""LAION sample embedding paths and GT mapping (caption suffix `_cap0`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from psi_slack_pkg._paths import REPO_ROOT


def _load_split(embed_dir: Path, dataset: str, split: str, modality: str):
    p = embed_dir / f"{dataset}_{split}_{modality}.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    d = np.load(p, allow_pickle=True)
    return d["embeddings"].astype(np.float32), np.asarray(d["ids"])


def load_laion_embeddings(
    dataset: str,
    backbone: str,
    direction: str,
    query_split: str,
    gallery_splits: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    embed_dir = REPO_ROOT / f"embeddings_{backbone}"
    if direction == "i2t":
        q_mod, g_mod = "image", "text"
    elif direction == "t2i":
        q_mod, g_mod = "text", "image"
    else:
        raise ValueError(f"unsupported direction for LAION: {direction}")

    q_arr, q_ids = _load_split(embed_dir, dataset, query_split, q_mod)
    g_chunks: list[np.ndarray] = []
    g_id_chunks: list[np.ndarray] = []
    for split in gallery_splits:
        a, ids = _load_split(embed_dir, dataset, split, g_mod)
        g_chunks.append(a)
        g_id_chunks.append(ids)
    g_arr = np.concatenate(g_chunks, axis=0)
    g_ids = np.concatenate(g_id_chunks, axis=0)

    q_emb = F.normalize(torch.from_numpy(q_arr).to(device), dim=1)
    g_emb = F.normalize(torch.from_numpy(g_arr).to(device), dim=1)
    return q_emb, g_emb, q_ids, g_ids


def build_pair_gt_mapping(q_ids: np.ndarray, g_ids: np.ndarray, direction: str) -> dict:
    g_id_to_idx: dict[str, list[int]] = {}
    for j, gid in enumerate(g_ids):
        s = str(gid)
        g_id_to_idx.setdefault(s, []).append(j)

    out: dict[str, list[int]] = {}
    if direction == "i2t":
        base_to_idx: dict[str, list[int]] = {}
        for j, gid in enumerate(g_ids):
            s = str(gid)
            base = s.rsplit("_cap", 1)[0] if "_cap" in s else s
            base_to_idx.setdefault(base, []).append(j)
        for qid in q_ids:
            s = str(qid)
            if s in base_to_idx:
                out[s] = base_to_idx[s]
    else:
        for qid in q_ids:
            s = str(qid)
            base = s.rsplit("_cap", 1)[0] if "_cap" in s else s
            if base in g_id_to_idx:
                out[s] = g_id_to_idx[base]
    return out
