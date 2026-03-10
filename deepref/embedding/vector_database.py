"""FAISS-backed vector database for storing and retrieving pre-computed embeddings."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Iterator, Literal

import faiss
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline


class VectorDatabase(Dataset):
    """FAISS-backed storage for embeddings + labels.

    Extends :class:`torch.utils.data.Dataset` for DataLoader compatibility.

    An optional preprocessing pipeline (StandardScaler → PCA(0.95) →
    L2 Normalizer) can be fitted via :meth:`fit_pipeline`.  Once fitted:

    * All stored embeddings are transformed in-place and the FAISS index is
      rebuilt with the reduced dimensionality.
    * Subsequent :meth:`add` calls accept raw embeddings of the original dim
      and apply the pipeline automatically before inserting.
    * :attr:`dim` reflects the PCA output dim; the original encoder dim is
      kept in :attr:`_raw_dim`.

    GPU acceleration is available when ``faiss-gpu`` is installed.  Pass
    ``device="cuda"`` (or ``"cuda:N"`` for a specific device) to move the
    FAISS index to GPU.  :meth:`save` always serialises a CPU copy of the
    index so files are portable.  :meth:`load` accepts the same ``device``
    argument to restore a GPU-resident index from a CPU file.

    Args:
        dim: embedding dimensionality of the raw encoder output.
        index_type: FAISS index type — ``"flat_l2"`` (L2 distance) or
            ``"flat_ip"`` (inner product / cosine when vectors are normalised).
        device: PyTorch-style device string — ``"cpu"`` (default), ``"cuda"``,
            or ``"cuda:N"`` for GPU device *N*.
    """

    _INDEX_SUFFIX = ".index"
    _LABELS_SUFFIX = ".labels.npy"
    _PIPELINE_SUFFIX = ".pipeline.pkl"

    def __init__(
        self,
        dim: int,
        index_type: Literal["flat_l2", "flat_ip"] = "flat_l2",
        device: str | None = "cpu",
    ) -> None:
        self.dim = dim          # current stored dim; updated after fit_pipeline
        self._raw_dim = dim     # original encoder output dim; stable across pipeline fit
        self._index_type = index_type
        self._device = device
        self._gpu_device_id: int | None = self._parse_gpu_device(device)
        self._gpu_res: Any = None
        self.pipeline: Pipeline | None = None
        self.index = self._build_index(dim)
        self._labels: np.ndarray = np.empty(0, dtype=np.int64)

    # ------------------------------------------------------------------
    # GPU helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_gpu_device(device: str | None) -> int | None:
        """Return the integer GPU device id, or ``None`` for CPU."""
        if not device or device == "cpu":
            return None
        if device == "cuda":
            return 0
        if device.startswith("cuda:"):
            return int(device.split(":", 1)[1])
        return None

    def _build_index(self, dim: int) -> faiss.Index:
        """Create a FAISS flat index of size *dim*, optionally on GPU."""
        if self._index_type == "flat_ip":
            cpu_index: faiss.Index = faiss.IndexFlatIP(dim)
        else:
            cpu_index = faiss.IndexFlatL2(dim)

        if self._gpu_device_id is None:
            return cpu_index

        # Move to GPU — keep resources alive as an instance attribute so the
        # GPU index is not invalidated by garbage collection.
        if self._gpu_res is None:
            self._gpu_res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(self._gpu_res, self._gpu_device_id, cpu_index)

    def _to_cpu_index(self) -> faiss.Index:
        """Return a CPU copy of the current index (no-op when already on CPU)."""
        if self._gpu_device_id is not None:
            return faiss.index_gpu_to_cpu(self.index)
        return self.index

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, embeddings: Tensor, labels: Tensor) -> None:
        """Append a batch of embeddings and their labels.

        If :meth:`fit_pipeline` has been called, *embeddings* must have the
        original raw dim (``_raw_dim``) and will be transformed automatically
        before insertion.  Otherwise, they must match ``dim``.

        Args:
            embeddings: float tensor of shape ``(N, expected_dim)``.
            labels: long tensor of shape ``(N,)``.

        Raises:
            ValueError: if shapes are inconsistent or dim does not match.
        """
        expected_dim = self._raw_dim if self.pipeline is not None else self.dim
        if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
            raise ValueError(
                f"embeddings must have shape (N, {expected_dim}), "
                f"got {tuple(embeddings.shape)}"
            )
        if labels.ndim != 1:
            raise ValueError(
                f"labels must have shape (N,), got {tuple(labels.shape)}"
            )
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError(
                f"batch size mismatch: embeddings has {embeddings.shape[0]} rows "
                f"but labels has {labels.shape[0]} elements"
            )

        np_emb = embeddings.detach().cpu().float().numpy().astype(np.float32)
        np_lbl = labels.detach().cpu().numpy().astype(np.int64)

        if self.pipeline is not None:
            np_emb = np.ascontiguousarray(
                self.pipeline.transform(np_emb).astype(np.float32)
            )

        self.index.add(np.ascontiguousarray(np_emb))
        self._labels = np.concatenate([self._labels, np_lbl])

    # ------------------------------------------------------------------
    # Preprocessing pipeline
    # ------------------------------------------------------------------

    def fit_pipeline(self) -> None:
        """Fit StandardScaler → PCA(0.95) → L2 Normalizer on stored embeddings.

        All embeddings currently in the FAISS index are read, the pipeline is
        fitted and applied, and the index is rebuilt with the reduced
        dimensionality.  After this call:

        * :attr:`dim` reflects the PCA output dim.
        * :attr:`_raw_dim` is unchanged (original encoder output dim).
        * :attr:`pipeline` holds the fitted :class:`sklearn.pipeline.Pipeline`.
        * Future :meth:`add` calls accept raw embeddings of dim ``_raw_dim``
          and apply the pipeline automatically.

        Raises:
            RuntimeError: if the database is empty.
        """
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import Normalizer, StandardScaler

        n = len(self)
        if n == 0:
            raise RuntimeError("Cannot fit pipeline on an empty database.")

        cpu_index = self._to_cpu_index()
        raw = np.stack([cpu_index.reconstruct(i) for i in range(n)])  # (N, raw_dim)

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            ("normalizer", Normalizer(norm="l2")),
        ])
        transformed = np.ascontiguousarray(
            self.pipeline.fit_transform(raw).astype(np.float32)
        )  # (N, D')

        new_dim = transformed.shape[1]
        self.index = self._build_index(new_dim)
        self.index.add(transformed)
        self.dim = new_dim

    def apply_pipeline(self, pipeline: "Pipeline") -> None:
        """Transform stored embeddings using an already-fitted *pipeline*.

        Unlike :meth:`fit_pipeline`, this method only *transforms* — it does
        not fit.  Use it to apply a pipeline that was fitted on the training
        set to a held-out test set so that both sets share the same projection.

        The FAISS index is rebuilt with the new (reduced) dimensionality, and
        :attr:`dim`, :attr:`_raw_dim`, and :attr:`pipeline` are updated to
        match :meth:`fit_pipeline` semantics so that subsequent :meth:`add`
        calls and :meth:`save` / :meth:`load` round-trips behave identically.

        Args:
            pipeline: a fitted :class:`sklearn.pipeline.Pipeline` with steps
                ``scaler``, ``pca``, and ``normalizer``.

        Raises:
            RuntimeError: if the database is empty.
        """
        n = len(self)
        if n == 0:
            raise RuntimeError("Cannot apply pipeline to an empty database.")

        cpu_index = self._to_cpu_index()
        raw = np.stack([cpu_index.reconstruct(i) for i in range(n)])  # (N, raw_dim)
        transformed = np.ascontiguousarray(
            pipeline.transform(raw).astype(np.float32)
        )  # (N, D')

        new_dim = transformed.shape[1]
        self.index = self._build_index(new_dim)
        self.index.add(transformed)
        self._raw_dim = raw.shape[1]
        self.dim = new_dim
        self.pipeline = pipeline

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.index.ntotal

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        cpu_index = self._to_cpu_index()
        vec = cpu_index.reconstruct(idx).copy()  # copy avoids dangling view
        embedding = torch.from_numpy(vec)         # (dim,) float32
        label = torch.tensor(self._labels[idx], dtype=torch.long)
        return embedding, label

    # ------------------------------------------------------------------
    # Convenience iterator
    # ------------------------------------------------------------------

    def iterate_batches(
        self,
        batch_size: int,
        shuffle: bool = False,
    ) -> Iterator[tuple[Tensor, Tensor]]:
        """Yield ``(embeddings, labels)`` tensor pairs of size *batch_size*.

        Args:
            batch_size: number of samples per batch.
            shuffle: if ``True``, samples are visited in random order.

        Yields:
            Tuples of ``(B, dim)`` float32 and ``(B,)`` long tensors.
        """
        n = len(self)
        cpu_index = self._to_cpu_index()
        indices = np.random.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            embs = np.stack([cpu_index.reconstruct(int(i)) for i in batch_idx])
            lbls = self._labels[batch_idx]
            yield torch.from_numpy(embs.copy()), torch.from_numpy(lbls.copy())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path_stem: str) -> None:
        """Write FAISS index, labels, and (if fitted) pipeline to disk.

        The index is always serialised as a CPU index so the file is portable
        across machines regardless of whether it was built on GPU.

        Args:
            path_stem: file path without extension.  Files written:
                ``<stem>.index``, ``<stem>.labels.npy``, and optionally
                ``<stem>.pipeline.pkl``.
        """
        import joblib

        faiss.write_index(self._to_cpu_index(), path_stem + self._INDEX_SUFFIX)
        np.save(path_stem + self._LABELS_SUFFIX, self._labels)
        if self.pipeline is not None:
            joblib.dump(self.pipeline, path_stem + self._PIPELINE_SUFFIX)

    @classmethod
    def load(cls, path_stem: str, device: str = "cpu") -> "VectorDatabase":
        """Load a :class:`VectorDatabase` from disk.

        Loads the FAISS index, labels, and the preprocessing pipeline if one
        was saved.  ``_raw_dim`` is recovered from the fitted scaler's
        ``n_features_in_`` attribute when a pipeline is present.

        Args:
            path_stem: file path without extension (same stem used in :meth:`save`).
            device: device to place the index on after loading — ``"cpu"``
                (default), ``"cuda"``, or ``"cuda:N"``.

        Returns:
            A fully populated :class:`VectorDatabase` instance.
        """
        import joblib

        cpu_index = faiss.read_index(path_stem + cls._INDEX_SUFFIX)
        labels = np.load(path_stem + cls._LABELS_SUFFIX)

        pipeline_path = path_stem + cls._PIPELINE_SUFFIX
        pipeline = joblib.load(pipeline_path) if os.path.exists(pipeline_path) else None

        obj = cls.__new__(cls)
        obj._index_type = "flat_l2"  # not persisted; default on load
        obj._labels = labels
        obj.pipeline = pipeline
        obj._device = device
        obj._gpu_device_id = cls._parse_gpu_device(device)
        obj._gpu_res = None

        if obj._gpu_device_id is not None:
            obj._gpu_res = faiss.StandardGpuResources()
            obj.index = faiss.index_cpu_to_gpu(
                obj._gpu_res, obj._gpu_device_id, cpu_index
            )
        else:
            obj.index = cpu_index

        obj.dim = cpu_index.d
        # Recover original encoder dim from the fitted scaler when available
        if pipeline is not None:
            obj._raw_dim = pipeline.named_steps["scaler"].n_features_in_
        else:
            obj._raw_dim = cpu_index.d
        return obj
