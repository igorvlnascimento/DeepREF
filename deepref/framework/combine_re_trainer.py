# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn, optim
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from deepref.dataset.combine_re_dataset import CombineREDataset
from deepref.dataset.re_dataset import REDataset
from deepref.framework.early_stopping import EarlyStopping
from deepref.framework.sentence_re_trainer import SentenceRETrainer
from deepref.framework.utils import AverageMeter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


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