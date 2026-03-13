"""Dataset pairing raw items (for on-the-fly encoder1) with cached encoder2 embeddings."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from deepref.dataset.combine_re_dataset import CombineREDataset
from deepref.embedding.vector_database import VectorDatabase


class HybridDataset(Dataset):
    """Pairs raw items with pre-computed encoder2 embeddings.

    Each ``__getitem__`` returns ``(item_dict, enc2_emb, label)`` so that
    :class:`~deepref.framework.hybrid_re_trainer.HybridRETrainer` can run
    encoder1 on-the-fly while fetching encoder2 embeddings from the VDB.

    The two sources must be aligned (same ordering, same length) — this is
    guaranteed when the VDB was generated from the same dataset with
    ``shuffle=False``.

    Args:
        base_dataset: the raw item dataset (e.g. :class:`CombineREDataset`).
        enc2_vdb: pre-computed encoder2 embeddings aligned with *base_dataset*.
    """

    def __init__(self, base_dataset: CombineREDataset, enc2_vdb: VectorDatabase) -> None:
        if len(base_dataset) != len(enc2_vdb):
            raise ValueError(
                f"base_dataset has {len(base_dataset)} samples but enc2_vdb has "
                f"{len(enc2_vdb)} entries — they must be equal."
            )
        self.base_dataset = base_dataset
        self.enc2_vdb = enc2_vdb

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[dict, torch.Tensor, torch.Tensor]:
        item, label = self.base_dataset[idx]
        enc2_emb, _ = self.enc2_vdb[idx]   # float32 (H2,)
        return item, enc2_emb, label


def hybrid_collate_fn(
    batch: list[tuple[dict, torch.Tensor, torch.Tensor]],
) -> dict[str, Any]:
    """Collate ``(item_dict, enc2_emb, label)`` triples into a batch dict."""
    items, enc2_embs, labels = zip(*batch)
    return {
        "labels": torch.stack(labels),
        "items": list(items),
        "enc2_embs": torch.stack(enc2_embs),
    }
