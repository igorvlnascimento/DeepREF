"""Hybrid trainer: encoder1 on-the-fly (trainable) + encoder2 from a pre-computed VDB."""

from __future__ import annotations

import logging
import os
from typing import Any

import mlflow
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from deepref.dataset.hybrid_dataset import HybridDataset, hybrid_collate_fn
from deepref.encoder.combine_embeddings import CombineEmbeddings
from deepref.framework.early_stopping import EarlyStopping
from deepref.framework.utils import AverageMeter
from deepref.model.softmax_mlp import SoftmaxMLP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


class HybridRETrainer(nn.Module):
    """Trains encoder1 end-to-end while encoder2 embeddings come from a pre-computed VDB.

    encoder1 is registered as a submodule so its parameters appear in
    ``self.named_parameters()`` and are covered by the optimizer.
    encoder2 is never called during training — its embeddings are read from
    the batch (pre-computed via :class:`~deepref.embedding.embedding_generator.EmbeddingGenerator`).

    Each training step:
    1. Runs encoder1 on the raw items (on-the-fly, with gradients).
    2. Fetches cached encoder2 embeddings from the batch.
    3. Concatenates → ``(B, H1+H2)``.
    4. Forwards through the MLP head and classifier.
    5. Back-propagates through encoder1 + MLP only.

    Args:
        encoder1: trainable encoder — will be optimised every step.
        model: :class:`SoftmaxMLP` whose MLP layers (``model.model``) and
            classifier head (``model.fc``) are also trained.  Its
            ``hidden_size`` must equal ``H1 + H2``.
        train_dataset: :class:`HybridDataset` for training.
        test_dataset: :class:`HybridDataset` for evaluation.
        ckpt: checkpoint file path.
        training_parameters: same dict accepted by :class:`CombineRETrainer`.
    """

    def __init__(
        self,
        encoder1: nn.Module,
        model: SoftmaxMLP,
        train_dataset: HybridDataset,
        test_dataset: HybridDataset,
        ckpt: str,
        training_parameters: dict[str, Any],
    ) -> None:
        super().__init__()

        # Register encoder1 as a submodule → its params appear in
        # self.named_parameters() and are covered by the optimizer.
        self.encoder1 = encoder1
        self.model = model
        self.training_parameters = training_parameters
        self.max_epoch = training_parameters["max_epoch"]
        self.criterion = training_parameters["criterion"]
        self.lr = training_parameters["lr"]
        batch_size = training_parameters["batch_size"]

        val_size = max(1, round(len(train_dataset) * 0.1))
        train_size = len(train_dataset) - val_size
        train_split, val_split = random_split(train_dataset, [train_size, val_size])
        logger.info(
            "Hybrid split — train: %d  val: %d  test: %d",
            train_size, val_size, len(test_dataset),
        )

        self.train_loader = DataLoader(
            train_split, batch_size=batch_size, shuffle=True,
            collate_fn=hybrid_collate_fn,
        )
        self.val_loader = DataLoader(
            val_split, batch_size=batch_size, shuffle=False,
            collate_fn=hybrid_collate_fn,
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=hybrid_collate_fn,
        )

        # Optimizer — encoder1 backbone params get their own LR/WD/warmup;
        # the MLP head uses the global training LR.
        opt = training_parameters["opt"]
        lr = training_parameters["lr"]
        weight_decay = training_parameters.get("weight_decay", 0.0)
        enc1_lr     = training_parameters.get("encoder1_lr", lr)
        enc1_wd     = training_parameters.get("encoder1_weight_decay", weight_decay)
        enc1_warmup = training_parameters.get("encoder1_warmup_step", 0)

        if opt in ("adam", "adamw"):
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            all_named = list(self.named_parameters())

            # encoder1 backbone params — registered under "encoder1._backbone_*"
            enc1_named = [
                (n, p) for n, p in all_named if "encoder1._backbone_" in n
            ]
            backbone_names = {n for n, _ in enc1_named}
            cls_named = [(n, p) for n, p in all_named if n not in backbone_names]

            def _make_groups(named_params, param_lr, param_wd, warmup_steps=0):
                decay  = [p for n, p in named_params if not any(nd in n for nd in no_decay)]
                no_dec = [p for n, p in named_params if     any(nd in n for nd in no_decay)]
                groups = []
                if decay:
                    groups.append({
                        "params": decay, "lr": param_lr, "weight_decay": param_wd,
                        "initial_lr": param_lr, "warmup_steps": warmup_steps,
                    })
                if no_dec:
                    groups.append({
                        "params": no_dec, "lr": param_lr, "weight_decay": 0.0,
                        "initial_lr": param_lr, "warmup_steps": warmup_steps,
                    })
                return groups

            param_groups = (
                _make_groups(enc1_named, enc1_lr, enc1_wd, enc1_warmup)
                + _make_groups(cls_named, lr, weight_decay)
            )
            Opt = torch.optim.AdamW if opt == "adamw" else torch.optim.Adam
            self.optimizer = Opt(param_groups)
        elif opt == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Invalid optimizer: {opt!r}. Choose sgd, adam, or adamw.")

        warmup_step = training_parameters.get("warmup_step", 0)
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = train_size // batch_size * self.max_epoch
            warmup_steps = int(warmup_step * training_steps)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=training_steps,
            )
        else:
            self.scheduler = None

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
        """One pass: run encoder1 on-the-fly, concat with cached enc2_embs, train MLP."""
        device = next(self.model.parameters()).device
        avg_loss = AverageMeter()
        avg_acc  = AverageMeter()

        t = tqdm(loader)
        for data in t:
            labels    = data["labels"].to(device)
            items     = data["items"]
            enc2_embs = F.normalize(data["enc2_embs"].to(device), dim=-1)

            # encoder1 on-the-fly — gradients flow through this call.
            enc1_embs = CombineEmbeddings._encode_batch(self.encoder1, items).to(device)
            emb = torch.cat([enc1_embs, enc2_embs], dim=-1)   # (B, H1+H2)

            rep    = self.model.model(emb)   # MLP layers  → (B, H')
            logits = self.model.fc(rep)      # classifier  → (B, N)
            _, pred = logits.max(-1)

            acc = float((pred == labels).long().sum()) / labels.size(0)
            avg_acc.update(acc, labels.size(0))

            if training:
                loss = self.criterion(logits, labels)
                avg_loss.update(loss.item(), 1)

                # Per-group linear warmup (manual, only when no HF scheduler).
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
        """Evaluate using hybrid batches (encoder1 on-the-fly + cached enc2_embs)."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        self.eval()
        pred_result: list[int] = []
        all_labels:  list[int] = []
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for data in tqdm(eval_loader):
                labels    = data["labels"].to(device)
                items     = data["items"]
                enc2_embs = F.normalize(data["enc2_embs"].to(device), dim=-1)

                enc1_embs = CombineEmbeddings._encode_batch(self.encoder1, items).to(device)
                emb = torch.cat([enc1_embs, enc2_embs], dim=-1)

                rep    = self.model.model(emb)
                logits = self.model.fc(rep)
                _, pred = logits.max(-1)
                pred_result.extend(pred.tolist())
                all_labels.extend(labels.tolist())

        result = {
            "acc":      accuracy_score(all_labels, pred_result),
            "micro_p":  precision_score(all_labels, pred_result, average="micro", zero_division=0),
            "micro_r":  recall_score(all_labels,   pred_result, average="micro", zero_division=0),
            "micro_f1": f1_score(all_labels,        pred_result, average="micro", zero_division=0),
            "macro_p":  precision_score(all_labels, pred_result, average="macro", zero_division=0),
            "macro_r":  recall_score(all_labels,    pred_result, average="macro", zero_division=0),
            "macro_f1": f1_score(all_labels,        pred_result, average="macro", zero_division=0),
        }
        logger.info("Eval result: %s", result)
        return result, pred_result, all_labels

    # ------------------------------------------------------------------
    # Training loop with MLflow logging
    # ------------------------------------------------------------------

    def train_model(self, warmup: bool = True, metric: str = "macro_f1") -> float:
        """Epoch loop with validation and early stopping.

        Returns:
            Best value of *metric* achieved on the validation split.
        """
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
