"""
Combine-Embeddings experiment runner.

Trains a SoftmaxMLP classifier on top of two concatenated encoders using
SentenceRETrainer as the training engine.

Four encoder combinations (driven by Hydra multirun):
  encoder1=relation  encoder2=bow_sdp
  encoder1=relation  encoder2=verbalized_sdp
  encoder1=llm       encoder2=bow_sdp
  encoder1=llm       encoder2=verbalized_sdp

Usage
-----
Single run (defaults: relation + bow_sdp, semeval2010):
    python deepref/experiments/run_combine_embeddings_experiments.py

All 4 encoder combinations via multirun:
    python deepref/experiments/run_combine_embeddings_experiments.py \\
        --multirun encoder1=relation,llm encoder2=bow_sdp,verbalized_sdp

Cross dataset × encoder multirun:
    python deepref/experiments/run_combine_embeddings_experiments.py \\
        --multirun \\
        dataset=semeval2010,ddi \\
        encoder1=relation,llm \\
        encoder2=bow_sdp,verbalized_sdp
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepref.dataset.re_dataset import REDataset
from deepref.encoder.llm_encoder import LLMEncoder
from deepref.encoder.relation_encoder import RelationEncoder
from deepref.encoder.sdp_encoder import BoWSDPEncoder, VerbalizedSDPEncoder
from deepref.framework.sentence_re_trainer import SentenceRETrainer
from deepref.framework.utils import AverageMeter
from deepref.model.softmax_mlp import SoftmaxMLP

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

    def __init__(self, csv_path: str, rel2id: dict | None = None) -> None:
        # Replicate only the DataFrame + rel2id parts of REDataset.__init__
        # to avoid requiring a HuggingFace tokenizer at the dataset level.
        self.df = self.get_dataframe(csv_path)
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

    train_ds = CombineREDataset(resolve(cfg_dataset.train_csv_path))
    test_ds = CombineREDataset(resolve(cfg_dataset.test_csv_path))

    # Unify relation labels across both splits
    all_relations = sorted(set(train_ds.rel2id) | set(test_ds.rel2id))
    unified_rel2id = {r: i for i, r in enumerate(all_relations)}
    train_ds.rel2id = unified_rel2id
    test_ds.rel2id = unified_rel2id

    logger.info(
        "Loaded '%s': train=%d  test=%d  classes=%d",
        cfg_dataset.name, len(train_ds), len(test_ds), len(unified_rel2id),
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
        # RelationEncoder sets self.hidden_size = 3 * model.config.hidden_size
        if isinstance(encoder, RelationEncoder):
            return encoder.hidden_size
        # BoWSDPEncoder has no neural model; output = dep_vocab length
        if isinstance(encoder, BoWSDPEncoder):
            return len(encoder.dep_vocab)
        # LLMEncoder (and VerbalizedSDPEncoder which inherits it)
        if isinstance(encoder, LLMEncoder):
            return encoder.model.config.hidden_size
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

        if isinstance(encoder, VerbalizedSDPEncoder):
            emb = encoder.forward(item)    # (1, H)
            return emb.squeeze(0).float()  # (H,)

        if isinstance(encoder, BoWSDPEncoder):
            return encoder.forward(item).float()  # (len_dep_vocab,)

        if isinstance(encoder, LLMEncoder):
            text = " ".join(item["token"])
            emb = encoder.forward(text)    # (1, H)
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
            emb2 = self._encode_single(self.encoder2, item)
            batch_embs.append(torch.cat([emb1, emb2], dim=0))
        return torch.stack(batch_embs, dim=0)  # (B, H1+H2)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class CombineRETrainer(SentenceRETrainer):
    """SentenceRETrainer adapted for combined-encoder experiments.

    Overrides three methods from the parent:

    * ``__init__`` — creates :class:`DataLoader` instances with
      :func:`combine_collate_fn` instead of RELoader, skipping the tokenizer
      requirement embedded in :class:`~deepref.dataset.re_dataset.RELoader`.
    * ``iterate_loader`` — handles ``dict`` batches ``{"labels": …, "items": …}``
      and moves only tensor data to the target device (the ``items`` list stays
      on CPU since encoder internals handle device placement).
    * ``eval_model`` — computes precision / recall / F1 directly with sklearn
      instead of delegating to ``REDataset.eval``, which cannot operate on a
      :class:`~torch.utils.data.Subset`.
    * ``train_model`` — adds per-epoch MLflow metric logging on top of the
      standard checkpointing logic.
    """

    def __init__(
        self,
        model: SoftmaxMLP,
        train_dataset: Subset,
        test_dataset: Subset,
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

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=combine_collate_fn,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=combine_collate_fn,
        )

        # Optimizer
        opt = training_parameters["opt"]
        lr = training_parameters["lr"]
        weight_decay = training_parameters.get("weight_decay", 0.0)

        params = self.parameters()
        if opt == "sgd":
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == "adam":
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == "adamw":
            from torch.optim import AdamW
            named = list(self.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            grouped = [
                {
                    "params": [p for n, p in named if not any(nd in n for nd in no_decay)],
                    "weight_decay": 0.01,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in named if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
            self.optimizer = AdamW(grouped)
        else:
            raise ValueError(f"Invalid optimizer: {opt!r}. Choose sgd, adam, or adamw.")

        # LR scheduler
        warmup_step = training_parameters.get("warmup_step", 0)
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = len(train_dataset) // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_step,
                num_training_steps=training_steps,
            )
        else:
            self.scheduler = None

        if torch.cuda.is_available():
            self.cuda()

        self.ckpt = ckpt

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
        global_step = 0

        t = tqdm(loader)
        for data in t:
            labels = data["labels"].to(device)
            items = data["items"]

            logits = self.model(items=items)
            _, pred = logits.max(-1)

            acc = float((pred == labels).long().sum()) / labels.size(0)
            avg_acc.update(acc, labels.size(0))
            t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)

            if training:
                loss = self.criterion(logits, labels)
                avg_loss.update(loss.item(), 1)

                if warmup:
                    warmup_steps = 300
                    rate = min(1.0, global_step / warmup_steps) if warmup_steps > 0 else 1.0
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.lr * rate

                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1

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
            "macro_f1": f1_score(all_labels, pred_result, average="macro", zero_division=0),
        }
        logger.info("Eval result: %s", result)
        return result, pred_result, all_labels

    # ------------------------------------------------------------------
    # Training loop with MLflow logging
    # ------------------------------------------------------------------

    def train_model(self, warmup: bool = True, metric: str = "micro_f1") -> float:
        """Train for ``max_epoch`` epochs, log metrics to MLflow per epoch.

        Saves the best checkpoint (by *metric* on the test set) to ``self.ckpt``.

        Returns:
            Best value of *metric* achieved during training.
        """
        best_metric = 0.0

        for epoch in range(self.max_epoch):
            logger.info("=== Epoch %d / %d — train ===", epoch + 1, self.max_epoch)
            self.train()
            self.iterate_loader(self.train_loader, warmup=warmup, training=True)

            logger.info("=== Epoch %d / %d — eval ===", epoch + 1, self.max_epoch)
            result, _, _ = self.eval_model(self.test_loader)

            logger.info(
                "Metric %s: current=%.4f  best=%.4f",
                metric, result[metric], best_metric,
            )
            mlflow.log_metrics(
                {f"val_{k}": v for k, v in result.items() if isinstance(v, float)},
                step=epoch,
            )

            if result[metric] > best_metric:
                logger.info("Best checkpoint — saving to %s", self.ckpt)
                folder = os.path.dirname(self.ckpt)
                if folder:
                    os.makedirs(folder, exist_ok=True)
                torch.save({"state_dict": self.model.state_dict()}, self.ckpt)
                best_metric = result[metric]

        logger.info("Best %s on test set: %.4f", metric, best_metric)
        return best_metric


# ---------------------------------------------------------------------------
# Encoder factory
# ---------------------------------------------------------------------------

def build_encoder1(cfg: DictConfig, device: str) -> nn.Module:
    """Instantiate the first encoder from Hydra config."""
    enc = cfg.encoder1
    if enc.type == "relation":
        return RelationEncoder(model_name=enc.model_name, max_length=enc.max_length)
    if enc.type == "llm":
        return LLMEncoder(
            model_name=enc.model_name,
            max_length=enc.max_length,
            device=device,
            trainable=enc.trainable,
        )
    raise ValueError(f"Unknown encoder1 type: {enc.type!r}")


def build_encoder2(cfg: DictConfig, device: str) -> nn.Module:
    """Instantiate the second encoder from Hydra config."""
    enc = cfg.encoder2
    if enc.type == "bow_sdp":
        return BoWSDPEncoder()
    if enc.type == "verbalized_sdp":
        return VerbalizedSDPEncoder(model_name=enc.model_name, device=device)
    raise ValueError(f"Unknown encoder2 type: {enc.type!r}")


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
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "dataset": cfg.dataset.name,
            "encoder1_type": cfg.encoder1.type,
            "encoder1_model": cfg.encoder1.get("model_name", "n/a"),
            "encoder2_type": cfg.encoder2.type,
            "encoder2_model": cfg.encoder2.get("model_name", "n/a"),
            "batch_size": cfg.training.batch_size,
            "lr": cfg.training.lr,
            "max_epoch": cfg.training.max_epoch,
            "num_mlp_layers": cfg.training.num_mlp_layers,
            "dropout": cfg.training.dropout,
            "opt": cfg.training.opt,
            "seed": cfg.training.seed,
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

            combine = CombineEmbeddings(encoder1, encoder2)

            # ── Model ───────────────────────────────────────────────────────
            model = SoftmaxMLP(
                sentence_encoder=combine,
                num_class=num_class,
                rel2id=train_dataset.rel2id,
                dropout=cfg.training.dropout,
                num_layers=cfg.training.num_mlp_layers,
            )

            # ── Trainer ─────────────────────────────────────────────────────
            ckpt_path = os.path.join(
                cwd,
                "ckpt",
                f"{cfg.dataset.name}_{cfg.encoder1.type}_{cfg.encoder2.type}.pth.tar",
            )

            training_parameters = {
                "max_epoch": cfg.training.max_epoch,
                "criterion": nn.CrossEntropyLoss(),
                "lr": cfg.training.lr,
                "batch_size": cfg.training.batch_size,
                "opt": cfg.training.opt,
                "weight_decay": cfg.training.get("weight_decay", 0.0),
                "warmup_step": cfg.training.get("warmup_step", 0),
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
            mlflow.set_tag("status", "success")

            logger.info("Run '%s' finished — best micro_f1: %.4f", run_name, best_micro_f1)

        except Exception:
            import traceback
            logger.error("Run '%s' failed:\n%s", run_name, traceback.format_exc())
            mlflow.set_tag("status", "failed")
            raise


if __name__ == "__main__":
    main()
