"""Tests for VectorDatabase."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from deepref.embedding.vector_database import VectorDatabase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vdb(dim: int = 16, n: int = 20) -> tuple[VectorDatabase, torch.Tensor, torch.Tensor]:
    vdb = VectorDatabase(dim)
    embeddings = torch.randn(n, dim)
    labels = torch.arange(n, dtype=torch.long)
    vdb.add(embeddings, labels)
    return vdb, embeddings, labels


# ---------------------------------------------------------------------------
# add / __len__ / __getitem__
# ---------------------------------------------------------------------------

class TestAddAndRetrieve:
    def test_add_stores_correct_count(self):
        vdb, _, _ = _make_vdb(dim=8, n=15)
        assert len(vdb) == 15

    def test_getitem_returns_correct_vector(self):
        dim = 16
        vdb, embeddings, labels = _make_vdb(dim=dim, n=10)
        for i in range(10):
            emb, lbl = vdb[i]
            assert emb.shape == (dim,)
            assert lbl.item() == labels[i].item()
            assert torch.allclose(emb, embeddings[i].float(), atol=1e-5)

    def test_add_multiple_batches(self):
        dim = 8
        vdb = VectorDatabase(dim)
        vdb.add(torch.randn(5, dim), torch.zeros(5, dtype=torch.long))
        vdb.add(torch.randn(7, dim), torch.ones(7, dtype=torch.long))
        assert len(vdb) == 12

    def test_getitem_returns_copy(self):
        """Reconstructed vector must be a copy, not a dangling view."""
        vdb, _, _ = _make_vdb(dim=4, n=3)
        emb, _ = vdb[0]
        original = emb.clone()
        emb[0] = 999.0
        emb2, _ = vdb[0]
        assert torch.allclose(emb2, original, atol=1e-5)


# ---------------------------------------------------------------------------
# add validation errors
# ---------------------------------------------------------------------------

class TestAddValidation:
    def test_raises_on_dim_mismatch(self):
        vdb = VectorDatabase(dim=16)
        with pytest.raises(ValueError, match="shape"):
            vdb.add(torch.randn(4, 8), torch.zeros(4, dtype=torch.long))

    def test_raises_on_batch_size_mismatch(self):
        vdb = VectorDatabase(dim=16)
        with pytest.raises(ValueError, match="batch size mismatch"):
            vdb.add(torch.randn(4, 16), torch.zeros(3, dtype=torch.long))

    def test_raises_on_wrong_embeddings_ndim(self):
        vdb = VectorDatabase(dim=16)
        with pytest.raises(ValueError, match="shape"):
            vdb.add(torch.randn(16), torch.zeros(1, dtype=torch.long))


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip_vectors_and_labels(self):
        dim = 32
        vdb, embeddings, labels = _make_vdb(dim=dim, n=25)

        with tempfile.TemporaryDirectory() as tmpdir:
            stem = os.path.join(tmpdir, "test_vdb")
            vdb.save(stem)

            loaded = VectorDatabase.load(stem)

        assert len(loaded) == 25
        assert loaded.dim == dim

        for i in range(25):
            emb, lbl = loaded[i]
            assert torch.allclose(emb, embeddings[i].float(), atol=1e-5)
            assert lbl.item() == labels[i].item()

    def test_save_creates_two_files(self):
        vdb, _, _ = _make_vdb(dim=8, n=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            stem = os.path.join(tmpdir, "vdb")
            vdb.save(stem)
            assert os.path.exists(stem + VectorDatabase._INDEX_SUFFIX)
            assert os.path.exists(stem + VectorDatabase._LABELS_SUFFIX)


# ---------------------------------------------------------------------------
# iterate_batches
# ---------------------------------------------------------------------------

class TestIterateBatches:
    def test_covers_all_samples_without_overlap(self):
        n = 17
        vdb, _, labels = _make_vdb(dim=8, n=n)
        seen_labels: list[int] = []
        for emb, lbl in vdb.iterate_batches(batch_size=5, shuffle=False):
            seen_labels.extend(lbl.tolist())
        assert sorted(seen_labels) == list(range(n))

    def test_correct_shapes(self):
        dim = 12
        vdb, _, _ = _make_vdb(dim=dim, n=10)
        for emb, lbl in vdb.iterate_batches(batch_size=4, shuffle=False):
            assert emb.ndim == 2
            assert emb.shape[1] == dim
            assert lbl.ndim == 1
            assert emb.shape[0] == lbl.shape[0]

    def test_shuffle_covers_all_samples(self):
        n = 20
        vdb, _, _ = _make_vdb(dim=8, n=n)
        seen: list[int] = []
        for _, lbl in vdb.iterate_batches(batch_size=6, shuffle=True):
            seen.extend(lbl.tolist())
        assert sorted(seen) == list(range(n))


# ---------------------------------------------------------------------------
# IndexFlatIP path
# ---------------------------------------------------------------------------

class TestIndexFlatIP:
    def test_flat_ip_instantiates_without_error(self):
        vdb = VectorDatabase(dim=16, index_type="flat_ip")
        emb = torch.randn(5, 16)
        lbl = torch.zeros(5, dtype=torch.long)
        vdb.add(emb, lbl)
        assert len(vdb) == 5


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

# Need enough samples and sufficiently high dim for PCA(0.95) to be stable
_PIPE_DIM = 64
_PIPE_N = 200


def _make_vdb_for_pipeline(dim: int = _PIPE_DIM, n: int = _PIPE_N):
    vdb = VectorDatabase(dim)
    embeddings = torch.randn(n, dim)
    labels = torch.arange(n, dtype=torch.long)
    vdb.add(embeddings, labels)
    return vdb, embeddings, labels


class TestFitPipeline:
    def test_dim_reduced_after_fit(self):
        vdb, _, _ = _make_vdb_for_pipeline()
        raw_dim = vdb.dim
        vdb.fit_pipeline()
        assert vdb.dim < raw_dim, "PCA should reduce dimensionality"

    def test_raw_dim_unchanged_after_fit(self):
        vdb, _, _ = _make_vdb_for_pipeline()
        vdb.fit_pipeline()
        assert vdb._raw_dim == _PIPE_DIM

    def test_pipeline_attribute_is_set(self):
        vdb, _, _ = _make_vdb_for_pipeline()
        assert vdb.pipeline is None
        vdb.fit_pipeline()
        assert vdb.pipeline is not None

    def test_pipeline_has_correct_steps(self):
        vdb, _, _ = _make_vdb_for_pipeline()
        vdb.fit_pipeline()
        step_names = list(vdb.pipeline.named_steps.keys())
        assert step_names == ["scaler", "pca", "normalizer"]

    def test_pca_variance_threshold(self):
        vdb, _, _ = _make_vdb_for_pipeline()
        vdb.fit_pipeline()
        pca = vdb.pipeline.named_steps["pca"]
        assert abs(pca.n_components - 0.95) < 1e-9

    def test_stored_count_unchanged_after_fit(self):
        vdb, _, _ = _make_vdb_for_pipeline()
        vdb.fit_pipeline()
        assert len(vdb) == _PIPE_N

    def test_embeddings_are_l2_normalised(self):
        vdb, _, _ = _make_vdb_for_pipeline()
        vdb.fit_pipeline()
        for i in range(0, _PIPE_N, 20):
            emb, _ = vdb[i]
            norm = emb.norm().item()
            assert abs(norm - 1.0) < 1e-5, f"sample {i} norm={norm:.6f}, expected 1.0"

    def test_raises_on_empty_database(self):
        vdb = VectorDatabase(dim=16)
        with pytest.raises(RuntimeError, match="empty"):
            vdb.fit_pipeline()

    def test_add_after_fit_accepts_raw_dim(self):
        vdb, _, _ = _make_vdb_for_pipeline()
        vdb.fit_pipeline()
        n_before = len(vdb)
        new_emb = torch.randn(5, _PIPE_DIM)  # raw dim
        new_lbl = torch.zeros(5, dtype=torch.long)
        vdb.add(new_emb, new_lbl)
        assert len(vdb) == n_before + 5

    def test_add_after_fit_rejects_wrong_dim(self):
        vdb, _, _ = _make_vdb_for_pipeline()
        vdb.fit_pipeline()
        with pytest.raises(ValueError):
            vdb.add(torch.randn(3, _PIPE_DIM // 2), torch.zeros(3, dtype=torch.long))

    def test_getitem_shape_after_fit(self):
        vdb, _, _ = _make_vdb_for_pipeline()
        vdb.fit_pipeline()
        emb, lbl = vdb[0]
        assert emb.shape == (vdb.dim,)
        assert lbl.ndim == 0

    def test_iterate_batches_shape_after_fit(self):
        vdb, _, _ = _make_vdb_for_pipeline()
        vdb.fit_pipeline()
        for emb, lbl in vdb.iterate_batches(batch_size=32):
            assert emb.shape[1] == vdb.dim
            assert lbl.ndim == 1
            break


class TestSaveLoadWithPipeline:
    def test_roundtrip_with_pipeline(self):
        vdb, _, labels = _make_vdb_for_pipeline()
        vdb.fit_pipeline()
        reduced_dim = vdb.dim

        with tempfile.TemporaryDirectory() as tmpdir:
            stem = os.path.join(tmpdir, "vdb_pipe")
            vdb.save(stem)
            assert os.path.exists(stem + VectorDatabase._PIPELINE_SUFFIX)

            loaded = VectorDatabase.load(stem)

        assert loaded.dim == reduced_dim
        assert loaded._raw_dim == _PIPE_DIM
        assert loaded.pipeline is not None

        # Vectors should be identical after reload
        for i in range(0, _PIPE_N, 20):
            emb_orig, lbl_orig = vdb[i]
            emb_load, lbl_load = loaded[i]
            assert torch.allclose(emb_orig, emb_load, atol=1e-5)
            assert lbl_orig.item() == lbl_load.item()

    def test_load_without_pipeline_file(self):
        vdb, _, _ = _make_vdb_for_pipeline()
        # no fit_pipeline call — pipeline should be None after load

        with tempfile.TemporaryDirectory() as tmpdir:
            stem = os.path.join(tmpdir, "vdb_nopipe")
            vdb.save(stem)
            loaded = VectorDatabase.load(stem)

        assert loaded.pipeline is None
        assert loaded._raw_dim == _PIPE_DIM

    def test_add_after_load_with_pipeline(self):
        vdb, _, _ = _make_vdb_for_pipeline()
        vdb.fit_pipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            stem = os.path.join(tmpdir, "vdb_add")
            vdb.save(stem)
            loaded = VectorDatabase.load(stem)

        n_before = len(loaded)
        loaded.add(torch.randn(4, _PIPE_DIM), torch.zeros(4, dtype=torch.long))
        assert len(loaded) == n_before + 4


# ---------------------------------------------------------------------------
# apply_pipeline
# ---------------------------------------------------------------------------

class TestApplyPipeline:
    """apply_pipeline() transfers a fitted pipeline from one VDB to another."""

    def _fitted_pipeline(self):
        """Return a pipeline fitted on random data of dim _PIPE_DIM."""
        vdb, _, _ = _make_vdb_for_pipeline()
        vdb.fit_pipeline()
        return vdb.pipeline

    def test_dim_reduced_after_apply(self):
        vdb, _, _ = _make_vdb_for_pipeline()
        pipeline = self._fitted_pipeline()
        vdb.apply_pipeline(pipeline)
        assert vdb.dim < _PIPE_DIM

    def test_dim_matches_fit_pipeline_output(self):
        """Both VDBs built from the same raw data must end up with the same dim."""
        raw = torch.randn(_PIPE_N, _PIPE_DIM)
        lbls = torch.arange(_PIPE_N, dtype=torch.long)

        train_vdb = VectorDatabase(_PIPE_DIM)
        train_vdb.add(raw, lbls)
        train_vdb.fit_pipeline()

        test_vdb = VectorDatabase(_PIPE_DIM)
        test_vdb.add(raw, lbls)
        test_vdb.apply_pipeline(train_vdb.pipeline)

        assert test_vdb.dim == train_vdb.dim

    def test_pipeline_attribute_set_after_apply(self):
        vdb, _, _ = _make_vdb_for_pipeline()
        pipeline = self._fitted_pipeline()
        assert vdb.pipeline is None
        vdb.apply_pipeline(pipeline)
        assert vdb.pipeline is pipeline

    def test_raw_dim_preserved_after_apply(self):
        vdb, _, _ = _make_vdb_for_pipeline()
        pipeline = self._fitted_pipeline()
        vdb.apply_pipeline(pipeline)
        assert vdb._raw_dim == _PIPE_DIM

    def test_embeddings_are_l2_normalised_after_apply(self):
        vdb, _, _ = _make_vdb_for_pipeline()
        pipeline = self._fitted_pipeline()
        vdb.apply_pipeline(pipeline)
        for i in range(0, _PIPE_N, 20):
            emb, _ = vdb[i]
            norm = emb.norm().item()
            assert abs(norm - 1.0) < 1e-5, f"sample {i} norm={norm:.6f}"

    def test_count_unchanged_after_apply(self):
        vdb, _, _ = _make_vdb_for_pipeline()
        pipeline = self._fitted_pipeline()
        vdb.apply_pipeline(pipeline)
        assert len(vdb) == _PIPE_N

    def test_raises_on_empty_database(self):
        vdb = VectorDatabase(dim=_PIPE_DIM)
        pipeline = self._fitted_pipeline()
        with pytest.raises(RuntimeError, match="empty"):
            vdb.apply_pipeline(pipeline)

    def test_save_load_roundtrip_after_apply(self):
        vdb, _, labels = _make_vdb_for_pipeline()
        pipeline = self._fitted_pipeline()
        vdb.apply_pipeline(pipeline)

        with tempfile.TemporaryDirectory() as tmpdir:
            stem = os.path.join(tmpdir, "apply_test")
            vdb.save(stem)
            loaded = VectorDatabase.load(stem)

        assert loaded.dim == vdb.dim
        assert loaded._raw_dim == _PIPE_DIM
        assert loaded.pipeline is not None
        for i in range(0, _PIPE_N, 20):
            emb_orig, _ = vdb[i]
            emb_load, _ = loaded[i]
            assert torch.allclose(emb_orig, emb_load, atol=1e-5)
