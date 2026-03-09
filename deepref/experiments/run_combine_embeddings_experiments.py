"""
Combine-Embeddings experiment runner.

Trains a classifier on top of one or two encoders using SentenceRETrainer as
the training engine.  When two encoders are used their embeddings are
concatenated via :class:`CombineEmbeddings`; when only one encoder is used a
:class:`SingleEncoderWrapper` feeds it directly into the classifier.

Supported classifiers (``training.model_type``):
  softmax_mlp   — MLP with softmax head (gradient-based, default)
  xgboost       — XGBoost multi-class classifier (sklearn API)
  lightgbm      — LightGBM multi-class classifier (sklearn API)

Dual-encoder combinations (driven by Hydra multirun):
  encoder1=relation  encoder2=bow_sdp
  encoder1=relation  encoder2=verbalized_sdp
  encoder1=llm       encoder2=bow_sdp
  encoder1=llm       encoder2=verbalized_sdp

Single-encoder mode (set encoder2=none):
  encoder1=relation  encoder2=none
  encoder1=llm       encoder2=none

Usage
-----
Single run (defaults: relation + bow_sdp, semeval2010):
    python deepref/experiments/run_combine_embeddings_experiments.py

All 4 dual-encoder combinations via multirun:
    python deepref/experiments/run_combine_embeddings_experiments.py \\
        --multirun encoder1=relation,llm encoder2=bow_sdp,verbalized_sdp

Single-encoder run (encoder1 only):
    python deepref/experiments/run_combine_embeddings_experiments.py \\
        encoder2=none

All single + dual combinations via multirun:
    python deepref/experiments/run_combine_embeddings_experiments.py \\
        --multirun encoder1=relation,llm encoder2=bow_sdp,verbalized_sdp,none

Cross dataset × encoder multirun:
    python deepref/experiments/run_combine_embeddings_experiments.py \\
        --multirun \\
        dataset=semeval2010,ddi \\
        encoder1=relation,llm \\
        encoder2=bow_sdp,verbalized_sdp,none

XGBoost / LightGBM run:
    python deepref/experiments/run_combine_embeddings_experiments.py \\
        training.model_type=xgboost
"""

from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import hydra
import mlflow
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from deepref.dataset.re_dataset import REDataset
from deepref.encoder.bert_entity_encoder import BertEntityEncoder
from deepref.encoder.llm_encoder import LLMEncoder
from deepref.encoder.relation_encoder import RelationEncoder
from deepref.encoder.sdp_encoder import BoWSDPEncoder, VerbalizedSDPEncoder
from deepref.framework.early_stopping import EarlyStopping
from deepref.framework.sentence_re_trainer import SentenceRETrainer
from deepref.framework.utils import AverageMeter
from deepref.model.softmax_mlp import SoftmaxMLP
from deepref.nlp.nlp_tool import NLPTool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CombineREDataset(REDataset):
    """REDataset subclass that returns raw item dicts instead of tokenized tensors.

    Each sample is returned as a ``(item_dict, label_tensor)`` pair where
    ``item_dict`` contains the keys expected by all supported encoders:

    * ``'token'``: list of str tokens from the original sentence.
    * ``'h'``: ``{'name': str, 'pos': [start, end_exclusive]}`` — head entity.
    * ``'t'``: ``{'name': str, 'pos': [start, end_exclusive]}`` — tail entity.

    Bypasses the tokenizer dependency of the base :class:`REDataset` since
    combined encoders perform their own tokenization internally.
    """

    def __init__(self, csv_path: str | list[str], rel2id: dict | None = None) -> None:
        # Replicate only the DataFrame + rel2id parts of REDataset.__init__
        # to avoid requiring a HuggingFace tokenizer at the dataset level.
        self.df = self.get_dataframe(csv_path)
        if self.df is not None:
            self.df = self.df.dropna(subset=["relation_type"]).reset_index(drop=True)
        self.rel2id = rel2id if rel2id is not None else self.get_labels_dict()
        self.pipeline = None
        self.max_length = 0

    def __getitem__(self, index: int) -> tuple[dict, torch.Tensor]:
        row = self.df.iloc[index]

        tokens: list[str] = row["original_sentence"].split()
        e1: dict = ast.literal_eval(row["e1"])
        e2: dict = ast.literal_eval(row["e2"])

        item = {
            "token": tokens,
            "h": {"name": e1["name"], "pos": e1["position"]},
            "t": {"name": e2["name"], "pos": e2["position"]},
        }
        label = torch.tensor(self.rel2id[row["relation_type"]], dtype=torch.long)
        return item, label


def combine_collate_fn(
    batch: list[tuple[dict, torch.Tensor]],
) -> dict[str, Any]:
    """Collate ``(item_dict, label)`` pairs into a batch dict.

    Returns a dict with:
    * ``"labels"``: ``(B,)`` long tensor — compatible with
      :meth:`SentenceRETrainer.iterate_loader`.
    * ``"items"``: plain Python list of item dicts — passed as the
      ``items`` keyword argument to :class:`CombineEmbeddings`.
    """
    items, labels = zip(*batch)
    return {"labels": torch.stack(labels), "items": list(items)}


def load_split_datasets(
    cfg_dataset: DictConfig,
    cwd: str,
) -> tuple[CombineREDataset, CombineREDataset]:
    """Load pre-split train and test datasets from the benchmark CSV files.

    Builds a unified ``rel2id`` from the union of both splits so that class
    indices are consistent across train and test.
    """
    def resolve(p: str) -> str:
        path = Path(p)
        return str(path if path.is_absolute() else Path(cwd) / path)

    train_paths = [resolve(cfg_dataset.train_csv_path)]
    for extra in cfg_dataset.get("extra_train_csv_paths", []):
        train_paths.append(resolve(extra))

    train_ds = CombineREDataset(train_paths if len(train_paths) > 1 else train_paths[0])
    test_ds = CombineREDataset(resolve(cfg_dataset.test_csv_path))

    # Unify relation labels across both splits
    all_relations = sorted(set(train_ds.rel2id) | set(test_ds.rel2id))
    unified_rel2id = {r: i for i, r in enumerate(all_relations)}
    train_ds.rel2id = unified_rel2id
    test_ds.rel2id = unified_rel2id

    logger.info(
        "Loaded '%s': train=%d (from %d file(s))  test=%d  classes=%d",
        cfg_dataset.name, len(train_ds), len(train_paths), len(test_ds), len(unified_rel2id),
    )
    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Combined encoder
# ---------------------------------------------------------------------------

class CombineEmbeddings(nn.Module):
    """Concatenate embeddings from two independent encoders.

    Supports any combination of:
    * :class:`~deepref.encoder.relation_encoder.RelationEncoder`
    * :class:`~deepref.encoder.llm_encoder.LLMEncoder`
    * :class:`~deepref.encoder.sdp_encoder.BoWSDPEncoder`
    * :class:`~deepref.encoder.sdp_encoder.VerbalizedSDPEncoder`

    The combined output dimension equals ``hidden_size(encoder1) +
    hidden_size(encoder2)`` and is exposed via ``self.model.config.hidden_size``
    so that :class:`~deepref.model.softmax_mlp.SoftmaxMLP` (and the underlying
    :class:`~deepref.module.nn.mlp.MLP`) can consume this encoder without
    modification.

    Args:
        encoder1: first encoder instance.
        encoder2: second encoder instance.
    """

    def __init__(self, encoder1: nn.Module, encoder2: nn.Module) -> None:
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2

        h1 = self._get_hidden_size(encoder1)
        h2 = self._get_hidden_size(encoder2)
        combined = h1 + h2

        # Expose model.config.hidden_size for SoftmaxMLP / MLP compatibility.
        self.model = SimpleNamespace(
            config=SimpleNamespace(hidden_size=combined)
        )
        logger.info(
            "CombineEmbeddings — encoder1: %d  encoder2: %d  combined: %d",
            h1, h2, combined,
        )

    # ------------------------------------------------------------------
    # Hidden-size helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_hidden_size(encoder: nn.Module) -> int:
        """Return the output dimensionality of *encoder* for a single sample."""
        # BertEntityEncoder (and its subclass RelationEncoder) expose self.hidden_size
        if isinstance(encoder, BertEntityEncoder):
            return encoder.hidden_size
        # BoWSDPEncoder has no neural model; output = dep_vocab length
        if isinstance(encoder, BoWSDPEncoder):
            return len(encoder.dep_vocab)
        # LLMEncoder (and VerbalizedSDPEncoder which inherits it)
        if isinstance(encoder, LLMEncoder):
            return encoder.registry.get_model_hidden_size(encoder.model_name)
        raise ValueError(
            f"Cannot determine hidden size for encoder type: {type(encoder)}"
        )

    # ------------------------------------------------------------------
    # Per-encoder encode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_single(encoder: nn.Module, item: dict) -> torch.Tensor:
        """Encode one sample and return a 1-D float32 embedding tensor.

        Each encoder has a different forward signature; this method dispatches
        appropriately.  Return shape is always ``(H,)`` so outputs can be
        concatenated with ``torch.cat``.

        .. note::
            :class:`RelationEncoder` and :class:`VerbalizedSDPEncoder` are
            checked *before* their parent :class:`LLMEncoder` because both
            are subclasses of it.
        """
        if isinstance(encoder, RelationEncoder):
            token, att_mask, pos_e1, pos_e2, pos_mask = encoder.tokenize(item)
            emb = encoder.forward(token, att_mask, pos_e1, pos_e2, pos_mask)
            return emb.squeeze(0).float()  # (3H,)

        if isinstance(encoder, BertEntityEncoder):
            token, att_mask, pos_e1, pos_e2 = encoder.tokenize(item)
            emb = encoder.forward(token, att_mask, pos_e1, pos_e2)
            return emb.squeeze(0).float()  # (2H,)

        if isinstance(encoder, VerbalizedSDPEncoder):
            emb = encoder.forward(item)    # (1, H)
            return emb.squeeze(0).float()  # (H,)

        if isinstance(encoder, BoWSDPEncoder):
            return encoder.forward(item).float()  # (len_dep_vocab,)

        if isinstance(encoder, LLMEncoder):
            token_ids, attention_mask = encoder.tokenize(item)
            emb = encoder.forward(token_ids, attention_mask)    # (1, H)
            return emb.squeeze(0).float()  # (H,)

        raise ValueError(f"Unsupported encoder type: {type(encoder)}")

    # ------------------------------------------------------------------
    # Forward — called as combine_emb(items=batch_list) by SoftmaxMLP
    # ------------------------------------------------------------------

    def forward(self, items: list[dict]) -> torch.Tensor:
        """Encode a batch of items and return concatenated embeddings.

        Args:
            items: list of item dicts with ``'token'``, ``'h'``, and ``'t'``
                keys (as returned by :class:`CombineREDataset`).

        Returns:
            Float32 tensor of shape ``(B, H1 + H2)``.
        """
        batch_embs: list[torch.Tensor] = []
        for item in items:
            emb1 = self._encode_single(self.encoder1, item)
            emb2 = self._encode_single(self.encoder2, item).to(emb1.device)
            batch_embs.append(torch.cat([emb1, emb2], dim=0))
        return torch.stack(batch_embs, dim=0)  # (B, H1+H2)


# ---------------------------------------------------------------------------
# Single-encoder wrapper
# ---------------------------------------------------------------------------

class SingleEncoderWrapper(nn.Module):
    """Thin wrapper that satisfies the :class:`SoftmaxMLP` interface for one encoder.

    Provides ``self.model.config.hidden_size`` (used by :class:`MLP` to
    compute layer widths) and a ``forward(items)`` method that dispatches via
    :meth:`CombineEmbeddings._encode_single`, so any encoder type supported by
    :class:`CombineEmbeddings` works here without duplication.

    Used automatically when ``encoder2.type == "none"`` (single-encoder mode).

    Args:
        encoder: the single encoder to wrap.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        hidden = CombineEmbeddings._get_hidden_size(encoder)
        self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=hidden))
        logger.info("SingleEncoderWrapper — hidden_size: %d", hidden)

    def forward(self, items: list[dict]) -> torch.Tensor:
        """Encode a batch and return a ``(B, H)`` float32 tensor."""
        batch_embs = [
            CombineEmbeddings._encode_single(self.encoder, item)
            for item in items
        ]
        return torch.stack(batch_embs, dim=0)


# ---------------------------------------------------------------------------
# Sklearn-based classifier wrapper
# ---------------------------------------------------------------------------

class SklearnREClassifier(nn.Module):
    """Wraps an XGBoost or LightGBM classifier with a sentence encoder.

    The encoder produces embeddings; the sklearn classifier is then fitted on
    those embeddings.  :meth:`forward` encodes a batch of items and returns
    class probabilities as a ``(B, C)`` float32 tensor, making it compatible
    with :meth:`CombineRETrainer.eval_model`.

    Training is performed by :meth:`CombineRETrainer._train_sklearn`, which
    collects all embeddings upfront and calls :meth:`fit` — no gradient
    descent is used.

    Args:
        sentence_encoder: combined encoder (``CombineEmbeddings`` or
            ``SingleEncoderWrapper``) that maps items to embeddings.
        num_class: number of relation classes.
        rel2id: relation-name → class-index mapping.
        model_type: ``"xgboost"`` or ``"lightgbm"``.
    """

    IS_SKLEARN: bool = True  # detected by CombineRETrainer

    def __init__(
        self,
        sentence_encoder: nn.Module,
        num_class: int,
        rel2id: dict,
        model_type: str = "xgboost",
    ) -> None:
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.rel2id = rel2id
        self.id2rel = {v: k for k, v in rel2id.items()}
        self._model_type = model_type
        self._clf = self._build_clf(model_type, num_class)

    @staticmethod
    def _build_clf(model_type: str, num_class: int):
        if model_type == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(
                objective="multi:softprob",
                num_class=num_class,
                eval_metric="mlogloss",
                n_estimators=100,
                use_label_encoder=False,
            )
        if model_type == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                objective="multiclass",
                num_class=num_class,
                n_estimators=100,
                verbose=-1,
            )
        raise ValueError(
            f"Unknown sklearn model_type: {model_type!r}. Choose 'xgboost' or 'lightgbm'."
        )

    def fit(self, X: "np.ndarray", y: "np.ndarray") -> None:  # type: ignore[name-defined]
        """Fit the sklearn classifier on precomputed embeddings."""
        self._clf.fit(X, y)

    def forward(self, items: list[dict]) -> torch.Tensor:
        """Encode *items* and return class probabilities as a ``(B, C)`` tensor.

        Compatible with ``logits.max(-1)`` used in
        :meth:`CombineRETrainer.eval_model`.
        """
        import numpy as np

        with torch.no_grad():
            embeddings = self.sentence_encoder(items=items)  # (B, H)
        X = embeddings.cpu().numpy()
        proba = self._clf.predict_proba(X)               # (B, C)
        return torch.from_numpy(np.array(proba, dtype=np.float32))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class CombineRETrainer(SentenceRETrainer):
    """SentenceRETrainer adapted for combined-encoder experiments.

    Overrides four methods from the parent:

    * ``__init__`` — splits *train_dataset* into a 90 % training split and a
      10 % validation split (minimum 1 sample), then creates
      :class:`DataLoader` instances with :func:`combine_collate_fn` for each
      split and for *test_dataset*, skipping the tokenizer requirement
      embedded in :class:`~deepref.dataset.re_dataset.RELoader`.
    * ``iterate_loader`` — handles ``dict`` batches ``{"labels": …, "items": …}``
      and moves only tensor data to the target device (the ``items`` list stays
      on CPU since encoder internals handle device placement).
    * ``eval_model`` — computes precision / recall / F1 directly with sklearn
      instead of delegating to ``REDataset.eval``, which cannot operate on a
      :class:`~torch.utils.data.Subset`.
    * ``train_model`` — evaluates on the validation split each epoch, logs
      metrics to MLflow, and saves the best checkpoint.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: REDataset,
        test_dataset: REDataset,
        ckpt: str,
        training_parameters: dict[str, Any],
    ) -> None:
        # Initialise nn.Module directly — we rebuild everything that
        # SentenceRETrainer.__init__ does but use our custom DataLoader.
        nn.Module.__init__(self)

        self.model = model
        self.training_parameters = training_parameters
        self.max_epoch = training_parameters["max_epoch"]
        self.criterion = training_parameters["criterion"]
        self.lr = training_parameters["lr"]
        batch_size = training_parameters["batch_size"]

        # Detect whether this is a sklearn-based model (XGBoost / LightGBM).
        # When True, the gradient-based optimizer and scheduler are skipped.
        self._is_sklearn: bool = getattr(model, "IS_SKLEARN", False)

        # Split train_dataset: 90 % train / 10 % validation (minimum 1 val sample).
        val_size = max(1, round(len(train_dataset) * 0.1))
        train_size = len(train_dataset) - val_size
        train_split, val_split = random_split(train_dataset, [train_size, val_size])
        logger.info(
            "Dataset split — train: %d  val: %d  test: %d",
            train_size, val_size, len(test_dataset),
        )

        self.train_loader = DataLoader(
            train_split,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=combine_collate_fn,
        )
        self.val_loader = DataLoader(
            val_split,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=combine_collate_fn,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=combine_collate_fn,
        )

        # Optimizer + scheduler — only for gradient-based models.
        if not self._is_sklearn:
            opt = training_parameters["opt"]
            lr = training_parameters["lr"]
            weight_decay = training_parameters.get("weight_decay", 0.0)
            enc1_lr     = training_parameters.get("encoder1_lr", lr)
            enc1_wd     = training_parameters.get("encoder1_weight_decay", weight_decay)
            enc1_warmup = training_parameters.get("encoder1_warmup_step", 0)
            enc2_lr     = training_parameters.get("encoder2_lr", lr)
            enc2_wd     = training_parameters.get("encoder2_weight_decay", weight_decay)
            enc2_warmup = training_parameters.get("encoder2_warmup_step", 0)

            if opt == "sgd":
                self.optimizer = optim.SGD(self.parameters(), lr, weight_decay=weight_decay)
            elif opt in ("adam", "adamw"):
                no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
                all_named = list(self.named_parameters())

                # encoder1 backbone: dual-encoder path (.encoder1._backbone_)
                # or single-encoder path (.encoder._backbone_)
                enc1_named = [
                    (n, p) for n, p in all_named
                    if "encoder1._backbone_" in n or "encoder._backbone_" in n
                ]
                # encoder2 backbone: dual-encoder path (.encoder2._backbone_)
                enc2_named = [
                    (n, p) for n, p in all_named
                    if "encoder2._backbone_" in n
                ]
                # classifier head: everything that is not a backbone
                backbone_names = {n for n, _ in enc1_named} | {n for n, _ in enc2_named}
                cls_named = [(n, p) for n, p in all_named if n not in backbone_names]

                def _make_groups(named_params, param_lr, param_wd, group_warmup_steps=0):
                    """Split params into decay / no-decay groups with stored initial_lr."""
                    decay  = [p for n, p in named_params if not any(nd in n for nd in no_decay)]
                    no_dec = [p for n, p in named_params if     any(nd in n for nd in no_decay)]
                    groups = []
                    if decay:
                        groups.append({"params": decay,  "lr": param_lr, "weight_decay": param_wd, "initial_lr": param_lr, "warmup_steps": group_warmup_steps})
                    if no_dec:
                        groups.append({"params": no_dec, "lr": param_lr, "weight_decay": 0.0,      "initial_lr": param_lr, "warmup_steps": group_warmup_steps})
                    return groups

                param_groups = (
                    _make_groups(enc1_named, enc1_lr, enc1_wd, enc1_warmup)
                    + _make_groups(enc2_named, enc2_lr, enc2_wd, enc2_warmup)
                    + _make_groups(cls_named,  lr,      weight_decay)
                )

                if opt == "adamw":
                    from torch.optim import AdamW
                    self.optimizer = AdamW(param_groups)
                else:
                    self.optimizer = optim.Adam(param_groups)
            else:
                raise ValueError(f"Invalid optimizer: {opt!r}. Choose sgd, adam, or adamw.")

            # LR scheduler
            warmup_step = training_parameters.get("warmup_step", 0)
            if warmup_step > 0:
                from transformers import get_linear_schedule_with_warmup
                training_steps = train_size // batch_size * self.max_epoch
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_step,
                    num_training_steps=training_steps,
                )
            else:
                self.scheduler = None
        else:
            # Sklearn models have no gradient-based optimizer.
            self.optimizer = None
            self.scheduler = None
            logger.info(
                "Sklearn model detected (%s) — skipping optimizer/scheduler setup.",
                getattr(model, "_model_type", "unknown"),
            )

        if torch.cuda.is_available():
            self.cuda()

        self.ckpt = ckpt
        self.global_step = 0

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def iterate_loader(
        self,
        loader: DataLoader,
        warmup: bool = True,
        global_step: int = 0,
        training: bool = True,
    ) -> float | None:
        """One pass through *loader* with dict-format ``{"labels", "items"}`` batches.

        Only tensor fields (``"labels"``) are moved to the model's device;
        the ``"items"`` list is kept as plain Python since each encoder's
        forward method handles its own device placement.

        Returns:
            Average training loss when ``training=True``; ``None`` otherwise.
        """
        device = next(self.model.parameters()).device
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()

        t = tqdm(loader)
        for data in t:
            labels = data["labels"].to(device)
            items = data["items"]

            logits = self.model(items=items)
            _, pred = logits.max(-1)

            acc = float((pred == labels).long().sum()) / labels.size(0)
            avg_acc.update(acc, labels.size(0))

            if training:
                loss = self.criterion(logits, labels)
                avg_loss.update(loss.item(), 1)

                # Manual linear warmup — only when no HF scheduler is active to
                # avoid both mechanisms fighting over the learning rate.
                # Each param group has its own warmup_steps and initial_lr so
                # per-encoder schedules are independent.
                if warmup and self.scheduler is None:
                    for pg in self.optimizer.param_groups:
                        group_warmup = pg.get("warmup_steps", 0)
                        if group_warmup > 0:
                            rate = min(1.0, (self.global_step + 1) / group_warmup)
                            pg["lr"] = pg.get("initial_lr", self.lr) * rate

                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)

        return avg_loss.avg if training else None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval_model(
        self,
        eval_loader: DataLoader,
    ) -> tuple[dict[str, float], list[int], list[int]]:
        """Evaluate on *eval_loader* and return metrics computed with sklearn.

        Overrides the parent to avoid calling ``loader.dataset.eval(pred_result)``,
        which does not work on :class:`~torch.utils.data.Subset` objects.

        Returns:
            ``(result_dict, pred_result, ground_truth)`` matching the
            signature expected by :meth:`train_model`.
        """
        self.eval()
        pred_result: list[int] = []
        all_labels: list[int] = []
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for data in tqdm(eval_loader):
                labels = data["labels"].to(device)
                items = data["items"]
                logits = self.model(items=items)
                _, pred = logits.max(-1)
                pred_result.extend(pred.tolist())
                all_labels.extend(labels.tolist())

        result = {
            "acc": accuracy_score(all_labels, pred_result),
            "micro_p": precision_score(all_labels, pred_result, average="micro", zero_division=0),
            "micro_r": recall_score(all_labels, pred_result, average="micro", zero_division=0),
            "micro_f1": f1_score(all_labels, pred_result, average="micro", zero_division=0),
            "macro_p": precision_score(all_labels, pred_result, average="macro", zero_division=0),
            "macro_r": recall_score(all_labels, pred_result, average="macro", zero_division=0),
            "macro_f1": f1_score(all_labels, pred_result, average="macro", zero_division=0),
        }
        logger.info("Eval result: %s", result)
        return result, pred_result, all_labels

    # ------------------------------------------------------------------
    # Sklearn helpers
    # ------------------------------------------------------------------

    def _collect_embeddings(
        self,
        loader: DataLoader,
    ) -> "tuple[np.ndarray, np.ndarray]":  # type: ignore[name-defined]
        """Collect all embeddings and labels from *loader* without gradients.

        Returns:
            ``(X, y)`` where ``X`` has shape ``(N, H)`` and ``y`` has shape
            ``(N,)``.  Uses the model's ``sentence_encoder`` directly so that
            the sklearn classifier head is bypassed.
        """
        import numpy as np

        self.eval()
        all_embeddings: list = []
        all_labels: list = []

        with torch.no_grad():
            for data in tqdm(loader, desc="Collecting embeddings"):
                items = data["items"]
                labels = data["labels"]
                embs = self.model.sentence_encoder(items=items)  # (B, H)
                all_embeddings.append(embs.cpu().numpy())
                all_labels.append(labels.numpy())

        return (
            np.concatenate(all_embeddings, axis=0),
            np.concatenate(all_labels, axis=0),
        )

    def _train_sklearn(self, metric: str) -> float:
        """Fit the sklearn classifier on encoder embeddings and return best *metric*.

        Steps:
        1. Encode all training samples (no gradient).
        2. Fit the sklearn model (``XGBClassifier`` / ``LGBMClassifier``).
        3. Save the fitted model with :mod:`joblib`.
        4. Evaluate on the validation split and log metrics to MLflow.
        """
        import joblib

        logger.info("=== Collecting train embeddings for sklearn fitting ===")
        X_train, y_train = self._collect_embeddings(self.train_loader)

        logger.info(
            "Fitting %s on %d samples (dim=%d) …",
            self.model._model_type, len(X_train), X_train.shape[1],
        )
        self.model.fit(X_train, y_train)

        # Persist the fitted sklearn estimator
        folder = os.path.dirname(self.ckpt)
        if folder:
            os.makedirs(folder, exist_ok=True)
        ckpt_sklearn = self.ckpt.replace(".pth", ".joblib")
        joblib.dump(self.model._clf, ckpt_sklearn)
        logger.info("Sklearn model saved to %s", ckpt_sklearn)

        # Validation metrics (single "epoch" for MLflow compatibility)
        logger.info("=== Val evaluation ===")
        result, _, _ = self.eval_model(self.val_loader)
        mlflow.log_metrics(
            {f"val_{k}": v for k, v in result.items() if isinstance(v, float)},
            step=0,
        )
        logger.info("Val result: %s", result)
        return result.get(metric, 0.0)

    # ------------------------------------------------------------------
    # Training loop with MLflow logging
    # ------------------------------------------------------------------

    def train_model(self, warmup: bool = True, metric: str = "macro_f1") -> float:
        """Train the model and return the best value of *metric*.

        Dispatches to :meth:`_train_sklearn` for sklearn-based models
        (XGBoost / LightGBM) or runs the gradient-based epoch loop for
        :class:`~deepref.model.softmax_mlp.SoftmaxMLP`.

        For gradient-based training:
        * Evaluates on the validation split after each epoch.
        * Saves the best checkpoint (by *metric*) to ``self.ckpt``.
        * Stops early when *metric* has not improved for ``patience``
          consecutive epochs (``patience=0`` disables early stopping).
        """
        if self._is_sklearn:
            return self._train_sklearn(metric)

        best_metric = 0.0
        patience = self.training_parameters.get("patience", 0)
        early_stopper = EarlyStopping(patience=patience)

        for epoch in range(self.max_epoch):
            logger.info("=== Epoch %d / %d — train ===", epoch + 1, self.max_epoch)
            self.train()
            self.iterate_loader(self.train_loader, warmup=warmup, training=True)

            logger.info("=== Epoch %d / %d — val ===", epoch + 1, self.max_epoch)
            result, _, _ = self.eval_model(self.val_loader)

            logger.info(
                "Metric %s: current=%.4f  best=%.4f",
                metric, result[metric], best_metric,
            )
            mlflow.log_metrics(
                {f"val_{k}": v for k, v in result.items() if isinstance(v, float)},
                step=epoch,
            )

            improved = result[metric] > best_metric
            if improved:
                logger.info("Best checkpoint — saving to %s", self.ckpt)
                folder = os.path.dirname(self.ckpt)
                if folder:
                    os.makedirs(folder, exist_ok=True)
                torch.save({"state_dict": self.model.state_dict()}, self.ckpt)
                best_metric = result[metric]

            if early_stopper.step(improved):
                logger.info(
                    "Early stopping triggered after epoch %d (no improvement for %d epochs)",
                    epoch + 1, patience,
                )
                break

        logger.info("Best %s on validation set: %.4f", metric, best_metric)
        return best_metric


# ---------------------------------------------------------------------------
# Encoder factory
# ---------------------------------------------------------------------------

def build_encoder1(cfg: DictConfig, device: str) -> nn.Module:
    """Instantiate the first encoder from Hydra config."""
    enc = cfg.encoder1
    if enc.type == "relation":
        return RelationEncoder(
            model_name=enc.model_name,
            max_length=enc.max_length,
            device=device,
            trainable=enc.get("trainable", False),
            attn_implementation=enc.get("attn_implementation", "eager"),
        )
    if enc.type == "bert_entity":
        return BertEntityEncoder(
            model_name=enc.model_name,
            max_length=enc.max_length,
            device=device,
            trainable=enc.get("trainable", False),
            attn_implementation=enc.get("attn_implementation", "eager"),
        )
    if enc.type == "llm":
        return LLMEncoder(
            model_name=enc.model_name,
            max_length=enc.max_length,
            device=device,
            trainable=enc.trainable,
            attn_implementation=enc.attn_implementation,
        )
    raise ValueError(f"Unknown encoder1 type: {enc.type!r}")


def build_nlp_tool(name: str) -> NLPTool:
    """Instantiate an NLP tool by name (``'spacy'`` or ``'stanza'``)."""
    if name == "spacy":
        from deepref.nlp.spacy_nlp_tool import SpacyNLPTool
        return SpacyNLPTool()
    if name == "stanza":
        from deepref.nlp.stanza_nlp_tool import StanzaNLPTool
        return StanzaNLPTool()
    raise ValueError(f"Unknown NLP tool: {name!r}. Choose 'spacy' or 'stanza'.")


def build_encoder2(cfg: DictConfig, device: str) -> nn.Module | None:
    """Instantiate the second encoder from Hydra config, or return ``None``.

    Returns ``None`` when ``cfg.encoder2.type == "none"``, enabling
    single-encoder mode via :class:`SingleEncoderWrapper` in
    :func:`build_combined_encoder`.
    """
    enc = cfg.encoder2
    if enc.type == "none":
        return None
    if enc.type == "bert_entity":
        return BertEntityEncoder(
            model_name=enc.model_name,
            max_length=enc.max_length,
            device=device,
            trainable=enc.get("trainable", False),
            attn_implementation=enc.get("attn_implementation", "eager"),
        )
    nlp_tool_name = enc.get("nlp_tool", "spacy")
    nlp_tool = build_nlp_tool(nlp_tool_name)
    if enc.type == "bow_sdp":
        return BoWSDPEncoder(nlp_tool=nlp_tool)
    if enc.type == "verbalized_sdp":
        return VerbalizedSDPEncoder(nlp_tool=nlp_tool, model_name=enc.model_name, device=device)
    raise ValueError(f"Unknown encoder2 type: {enc.type!r}")


def build_combined_encoder(
    encoder1: nn.Module,
    encoder2: nn.Module | None,
) -> nn.Module:
    """Return a :class:`CombineEmbeddings` or :class:`SingleEncoderWrapper`.

    Args:
        encoder1: always required.
        encoder2: second encoder, or ``None`` for single-encoder mode.

    Returns:
        :class:`CombineEmbeddings` when both encoders are provided;
        :class:`SingleEncoderWrapper` wrapping *encoder1* when *encoder2*
        is ``None``.
    """
    if encoder2 is None:
        logger.info("Single-encoder mode — wrapping encoder1 with SingleEncoderWrapper")
        return SingleEncoderWrapper(encoder1)
    logger.info("Dual-encoder mode — combining encoders with CombineEmbeddings")
    return CombineEmbeddings(encoder1, encoder2)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    cfg: DictConfig,
    sentence_encoder: nn.Module,
    num_class: int,
    rel2id: dict,
) -> nn.Module:
    """Instantiate the classifier from ``cfg.training.model_type``.

    Args:
        cfg: full Hydra config.
        sentence_encoder: combined encoder to embed sentences.
        num_class: number of relation classes.
        rel2id: relation-name → class-index mapping.

    Returns:
        A :class:`~deepref.model.softmax_mlp.SoftmaxMLP` for
        ``model_type="softmax_mlp"`` or a :class:`SklearnREClassifier`
        for ``"xgboost"`` / ``"lightgbm"``.
    """
    model_type: str = cfg.training.get("model_type", "softmax_mlp")

    if model_type == "softmax_mlp":
        return SoftmaxMLP(
            sentence_encoder=sentence_encoder,
            num_class=num_class,
            rel2id=rel2id,
            dropout=cfg.training.dropout,
            num_layers=cfg.training.num_mlp_layers,
        )

    if model_type in ("xgboost", "lightgbm"):
        return SklearnREClassifier(
            sentence_encoder=sentence_encoder,
            num_class=num_class,
            rel2id=rel2id,
            model_type=model_type,
        )

    raise ValueError(
        f"Unknown model_type: {model_type!r}. "
        "Choose 'softmax_mlp', 'xgboost', or 'lightgbm'."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    rel2id: dict[str, int],
    run_name: str,
) -> None:
    """Plot a confusion matrix and log it as an MLflow PNG artifact.

    The figure is written to a temporary file, logged under the MLflow
    ``figures/`` artifact path, then deleted.

    Args:
        y_true:   ground-truth class indices.
        y_pred:   predicted class indices.
        rel2id:   relation-name → class-index mapping (used to derive labels).
        run_name: used as the figure title.
    """
    import tempfile

    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe in all environments
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    id2rel = {v: k for k, v in rel2id.items()}
    labels = sorted(id2rel.keys())
    class_names = [id2rel[i] for i in labels]
    n = len(labels)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Scale figure so each cell is roughly 0.6 in × 0.6 in; minimum 8 × 8.
    fig_size = max(8, n * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm,
        annot=n <= 30,   # skip per-cell numbers when the matrix is very large
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.4,
        ax=ax,
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(f"Confusion matrix — {run_name}", fontsize=13, pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=max(6, 10 - n // 5))
    plt.yticks(rotation=0,  fontsize=max(6, 10 - n // 5))
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig_path = f.name
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    mlflow.log_artifact(fig_path, artifact_path="figures")
    os.remove(fig_path)
    logger.info("Confusion matrix logged to MLflow artifacts (figures/)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="combine_experiment",
)
def main(cfg: DictConfig) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.training.seed)
    device: str = cfg.device

    # ── MLflow ──────────────────────────────────────────────────────────────
    if cfg.mlflow.tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    run_name = (
        f"{cfg.dataset.name}"
        f"__{cfg.encoder1.type}"
        f"__{cfg.encoder2.type}"
        f"__{cfg.training.get('model_type', 'softmax_mlp')}"
    )

    with mlflow.start_run(run_name=run_name):
        # Per-encoder training overrides (fall back to global training params)
        enc1_training = cfg.encoder1.get("training", {})
        enc2_training = cfg.encoder2.get("training", {}) if cfg.encoder2.type != "none" else {}
        enc1_lr     = enc1_training.get("lr",           cfg.training.lr)                    if enc1_training else cfg.training.lr
        enc1_wd     = enc1_training.get("weight_decay", cfg.training.get("weight_decay", 0.0)) if enc1_training else cfg.training.get("weight_decay", 0.0)
        enc1_warmup = enc1_training.get("warmup_step",  0)                                  if enc1_training else 0
        enc2_lr     = enc2_training.get("lr",           cfg.training.lr)                    if enc2_training else cfg.training.lr
        enc2_wd     = enc2_training.get("weight_decay", cfg.training.get("weight_decay", 0.0)) if enc2_training else cfg.training.get("weight_decay", 0.0)
        enc2_warmup = enc2_training.get("warmup_step",  0)                                  if enc2_training else 0

        model_type: str = cfg.training.get("model_type", "softmax_mlp")

        mlflow.log_params({
            "dataset": cfg.dataset.name,
            "model_type": model_type,
            "encoder1_type": cfg.encoder1.type,
            "encoder1_model": cfg.encoder1.get("model_name", "n/a"),
            "encoder1_trainable": cfg.encoder1.get("trainable", False),
            "encoder1_lr": enc1_lr,
            "encoder1_weight_decay": enc1_wd,
            "encoder1_warmup_step": enc1_warmup,
            "encoder2_type": cfg.encoder2.type,
            "encoder2_model": cfg.encoder2.get("model_name", "n/a"),
            "encoder2_nlp_tool": cfg.encoder2.get("nlp_tool", "n/a"),
            "encoder2_trainable": cfg.encoder2.get("trainable", False),
            "encoder2_lr": enc2_lr if cfg.encoder2.type != "none" else "n/a",
            "encoder2_weight_decay": enc2_wd if cfg.encoder2.type != "none" else "n/a",
            "encoder2_warmup_step": enc2_warmup if cfg.encoder2.type != "none" else "n/a",
            "batch_size": cfg.training.batch_size,
            "lr": cfg.training.lr,
            "max_epoch": cfg.training.max_epoch,
            "num_mlp_layers": cfg.training.get("num_mlp_layers", "n/a"),
            "dropout": cfg.training.get("dropout", "n/a"),
            "opt": cfg.training.opt,
            "seed": cfg.training.seed,
            "patience": cfg.training.get("patience", 0),
        })

        try:
            # ── Dataset ─────────────────────────────────────────────────────
            cwd = hydra.utils.get_original_cwd()
            train_dataset, test_dataset = load_split_datasets(cfg.dataset, cwd)
            num_class = len(train_dataset.rel2id)

            # ── Encoders ────────────────────────────────────────────────────
            logger.info("Building encoder1 (%s) …", cfg.encoder1.type)
            encoder1 = build_encoder1(cfg, device)

            logger.info("Building encoder2 (%s) …", cfg.encoder2.type)
            encoder2 = build_encoder2(cfg, device)

            combine = build_combined_encoder(encoder1, encoder2)

            # ── Model ───────────────────────────────────────────────────────
            model = build_model(cfg, combine, num_class, train_dataset.rel2id)
            logger.info("Classifier: %s", model_type)

            # ── Trainer ─────────────────────────────────────────────────────
            enc1_model_name = cfg.encoder1.get("model_name", cfg.encoder1.type)
            ckpt_dir = os.path.join(
                cwd,
                "ckpt",
                enc1_model_name.replace("/", "_"),
            )
            ckpt_path = os.path.join(
                ckpt_dir,
                f"{cfg.dataset.name}_{cfg.encoder1.type}_{cfg.encoder2.type}_{model_type}.pth",
            )

            training_parameters = {
                "max_epoch": cfg.training.max_epoch,
                "criterion": nn.CrossEntropyLoss(),
                "lr": cfg.training.lr,
                "batch_size": cfg.training.batch_size,
                "opt": cfg.training.opt,
                "weight_decay": cfg.training.get("weight_decay", 0.0),
                "warmup_step": cfg.training.get("warmup_step", 0),
                "patience": cfg.training.get("patience", 0),
                "encoder1_lr": enc1_lr,
                "encoder1_weight_decay": enc1_wd,
                "encoder1_warmup_step": enc1_warmup,
                "encoder2_lr": enc2_lr,
                "encoder2_weight_decay": enc2_wd,
                "encoder2_warmup_step": enc2_warmup,
            }

            trainer = CombineRETrainer(
                model=model,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                ckpt=ckpt_path,
                training_parameters=training_parameters,
            )

            # ── Training ────────────────────────────────────────────────────
            best_micro_f1 = trainer.train_model(metric="micro_f1")

            mlflow.log_metric("best_micro_f1", best_micro_f1)

            # ── Test evaluation ─────────────────────────────────────────────
            logger.info("=== Test evaluation ===")
            test_result, test_preds, test_labels = trainer.eval_model(trainer.test_loader)
            mlflow.log_metrics({
                "test_micro_p":  test_result["micro_p"],
                "test_micro_r":  test_result["micro_r"],
                "test_micro_f1": test_result["micro_f1"],
                "test_macro_p":  test_result["macro_p"],
                "test_macro_r":  test_result["macro_r"],
                "test_macro_f1": test_result["macro_f1"],
            })
            logger.info(
                "Test metrics — micro_f1: %.4f  macro_f1: %.4f",
                test_result["micro_f1"], test_result["macro_f1"],
            )

            _log_confusion_matrix(test_labels, test_preds, train_dataset.rel2id, run_name)

            mlflow.set_tag("status", "success")

            logger.info("Run '%s' finished — best micro_f1: %.4f", run_name, best_micro_f1)

        except Exception:
            import traceback
            logger.error("Run '%s' failed:\n%s", run_name, traceback.format_exc())
            mlflow.set_tag("status", "failed")
            raise


if __name__ == "__main__":
    main()
