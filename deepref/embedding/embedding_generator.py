"""Batch embedding generator that populates a VectorDatabase."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from deepref.embedding.vector_database import VectorDatabase


class EmbeddingGenerator:
    """Run an encoder over a dataset in batch mode and populate a :class:`VectorDatabase`.

    Decouples the (slow, GPU-bound) encoding step from the (fast, CPU-viable)
    classification training step, enabling faster iteration on MLP heads
    without re-running transformers.

    Args:
        encoder: a :class:`CombineEmbeddings` or :class:`SingleEncoderWrapper`
            instance.  Must expose ``model.config.hidden_size``.
        dataset: source dataset whose items are fed to the encoder.
        batch_size: number of samples per forward pass.
        device: device string passed to the encoder (unused directly here;
            the encoder handles its own device placement).
        index_type: FAISS index type forwarded to :class:`VectorDatabase`.
        collate_fn: collate function passed to the DataLoader.  Must produce
            a batch dict with ``"items"`` (list of item dicts) and ``"labels"``
            (long tensor).  Defaults to ``None`` (PyTorch default collation).
    """

    def __init__(
        self,
        encoder: nn.Module,
        dataset: Dataset,
        batch_size: int = 64,
        device: str = "cpu",
        faiss_device: str = "cpu",
        index_type: str = "flat_l2",
        collate_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.encoder = encoder
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.faiss_device = faiss_device
        self.index_type = index_type
        self.collate_fn = collate_fn
        self.embedding_dim: int = encoder.model.config.hidden_size

    def generate(self) -> VectorDatabase:
        """Encode all dataset samples and return a populated :class:`VectorDatabase`.

        The encoder is switched to eval mode during generation and restored to
        its original mode afterwards.

        Returns:
            A :class:`VectorDatabase` containing all embeddings and labels.
        """
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

        was_training = self.encoder.training
        self.encoder.eval()

        all_embeddings: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        n_batches = len(loader)
        try:
            with torch.no_grad():
                for batch in tqdm(loader, total=n_batches, desc="Generating embeddings", unit="batch"):
                    embeddings = self.encoder(items=batch["items"])  # (B, H)
                    all_embeddings.append(embeddings.cpu().float())
                    all_labels.append(batch["labels"].cpu())
        finally:
            if was_training:
                self.encoder.train()

        vdb = VectorDatabase(self.embedding_dim, self.index_type, device=self.faiss_device)
        vdb.add(torch.cat(all_embeddings), torch.cat(all_labels))
        return vdb
