"""Tests for EmbeddingGenerator."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch.nn as nn
from torch.utils.data import Dataset

import torch

from deepref.embedding.embedding_generator import EmbeddingGenerator
from deepref.embedding.vector_database import VectorDatabase


def _collate_fn(batch):
    items, labels = zip(*batch)
    return {"items": list(items), "labels": torch.stack(labels)}


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

HIDDEN = 32
N_SAMPLES = 12


class _FixedEncoder(nn.Module):
    """Stub encoder that returns deterministic fixed embeddings."""

    def __init__(self, hidden_size: int = HIDDEN) -> None:
        super().__init__()
        self.model = SimpleNamespace(
            config=SimpleNamespace(hidden_size=hidden_size)
        )
        # Trainable param so nn.Module.training flag is meaningful
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, items: list[dict]) -> torch.Tensor:
        """Return a fixed tensor of shape (B, hidden_size)."""
        B = len(items)
        return torch.ones(B, self.model.config.hidden_size)


class _StubDataset(Dataset):
    """In-memory dataset returning (item_dict, label_tensor) pairs."""

    def __init__(self, n: int = N_SAMPLES) -> None:
        self.n = n
        self.items = [
            {
                "token": ["word"],
                "h": {"name": "h", "pos": [0, 1]},
                "t": {"name": "t", "pos": [0, 1]},
            }
            for _ in range(n)
        ]
        self.labels = list(range(n))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[dict, torch.Tensor]:
        return self.items[idx], torch.tensor(self.labels[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def encoder():
    return _FixedEncoder(HIDDEN)


@pytest.fixture()
def dataset():
    return _StubDataset(N_SAMPLES)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmbeddingGeneratorGenerate:
    def test_returns_vector_database(self, encoder, dataset):
        gen = EmbeddingGenerator(encoder, dataset, batch_size=4, collate_fn=_collate_fn)
        vdb = gen.generate()
        assert isinstance(vdb, VectorDatabase)

    def test_vdb_length_equals_dataset_size(self, encoder, dataset):
        gen = EmbeddingGenerator(encoder, dataset, batch_size=4, collate_fn=_collate_fn)
        vdb = gen.generate()
        assert len(vdb) == N_SAMPLES

    def test_vdb_dim_matches_hidden_size(self, encoder, dataset):
        gen = EmbeddingGenerator(encoder, dataset, batch_size=4, collate_fn=_collate_fn)
        vdb = gen.generate()
        assert vdb.dim == HIDDEN

    def test_embeddings_have_correct_shape(self, encoder, dataset):
        gen = EmbeddingGenerator(encoder, dataset, batch_size=4, collate_fn=_collate_fn)
        vdb = gen.generate()
        emb, _ = vdb[0]
        assert emb.shape == (HIDDEN,)

    def test_labels_are_stored_correctly(self, encoder, dataset):
        gen = EmbeddingGenerator(encoder, dataset, batch_size=4, collate_fn=_collate_fn)
        vdb = gen.generate()
        for i in range(N_SAMPLES):
            _, lbl = vdb[i]
            assert lbl.item() == i

    def test_train_mode_restored_after_generate(self, encoder, dataset):
        encoder.train()
        assert encoder.training

        gen = EmbeddingGenerator(encoder, dataset, batch_size=4, collate_fn=_collate_fn)
        gen.generate()

        assert encoder.training, "encoder should be back in train mode"

    def test_eval_mode_stays_eval_after_generate(self, encoder, dataset):
        encoder.eval()
        assert not encoder.training

        gen = EmbeddingGenerator(encoder, dataset, batch_size=4, collate_fn=_collate_fn)
        gen.generate()

        assert not encoder.training, "encoder was already eval and should remain so"

    def test_generate_with_batch_larger_than_dataset(self, encoder, dataset):
        gen = EmbeddingGenerator(encoder, dataset, batch_size=N_SAMPLES * 2, collate_fn=_collate_fn)
        vdb = gen.generate()
        assert len(vdb) == N_SAMPLES

    def test_embedding_dim_attribute(self, encoder, dataset):
        gen = EmbeddingGenerator(encoder, dataset, batch_size=4, collate_fn=_collate_fn)
        assert gen.embedding_dim == HIDDEN
