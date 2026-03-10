# ---------------------------------------------------------------------------
# VectorDB trainer
# ---------------------------------------------------------------------------

import logging
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn, optim
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from deepref.embedding.vector_database import VectorDatabase
from deepref.framework.combine_re_trainer import CombineRETrainer
from deepref.framework.utils import AverageMeter
from deepref.model.softmax_mlp import SoftmaxMLP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

class VectorDBRETrainer(CombineRETrainer):
    """CombineRETrainer variant that trains directly on pre-computed embeddings.

    Accepts :class:`~deepref.embedding.vector_database.VectorDatabase` instances
    instead of :class:`CombineREDataset`.  The encoder is bypassed at training
    time: each batch is an ``(embeddings, labels)`` tensor pair produced by the
    VDB DataLoader, and the forward pass calls the MLP head and classifier
    directly — ``model.model(emb)`` → ``model.fc(rep)`` — skipping
    ``model.sentence_encoder``.

    This enables fast iteration on the classifier head without re-running the
    (slow, GPU-bound) encoder on every epoch.

    Args:
        model: :class:`~deepref.model.softmax_mlp.SoftmaxMLP` instance.
        train_vdb: pre-computed training embeddings.
        test_vdb: pre-computed test embeddings.
        ckpt: checkpoint file path.
        training_parameters: same dict accepted by :class:`CombineRETrainer`.
    """

    def __init__(
        self,
        model: SoftmaxMLP,
        train_vdb: VectorDatabase,
        test_vdb: VectorDatabase,
        ckpt: str,
        training_parameters: dict[str, Any],
    ) -> None:
        nn.Module.__init__(self)

        self.model = model
        self.training_parameters = training_parameters
        self._is_sklearn: bool = getattr(model, "IS_SKLEARN", False)
        self.max_epoch = training_parameters["max_epoch"]
        self.criterion = training_parameters["criterion"]
        self.lr = training_parameters["lr"]
        batch_size = training_parameters["batch_size"]

        val_size = max(1, round(len(train_vdb) * 0.1))
        train_size = len(train_vdb) - val_size
        train_split, val_split = random_split(train_vdb, [train_size, val_size])
        logger.info(
            "VectorDB split — train: %d  val: %d  test: %d",
            train_size, val_size, len(test_vdb),
        )

        self.train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_split,   batch_size=batch_size, shuffle=False)
        self.test_loader  = DataLoader(test_vdb,    batch_size=batch_size, shuffle=False)

        opt = training_parameters["opt"]
        weight_decay = training_parameters.get("weight_decay", 0.0)
        if opt == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(), self.lr, weight_decay=weight_decay
            )
        elif opt in ("adam", "adamw"):
            from torch.optim import AdamW
            Opt = AdamW if opt == "adamw" else optim.Adam
            self.optimizer = Opt(
                model.parameters(), lr=self.lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Invalid optimizer: {opt!r}. Choose sgd, adam, or adamw.")

        self.scheduler = None

        if torch.cuda.is_available():
            self.cuda()

        self.ckpt = ckpt
        self.global_step = 0

    def iterate_loader(
        self,
        loader: DataLoader,
        warmup: bool = True,
        global_step: int = 0,
        training: bool = True,
    ) -> float | None:
        """One pass through a VDB DataLoader with ``(embeddings, labels)`` batches.

        The encoder is not invoked.  Embeddings are forwarded directly through
        ``model.model`` (MLP layers) and ``model.fc`` (classifier head).
        """
        device = next(self.model.parameters()).device
        avg_loss = AverageMeter()
        avg_acc  = AverageMeter()

        t = tqdm(loader)
        for emb, labels in t:
            emb    = emb.to(device)
            labels = labels.to(device)

            rep    = self.model.model(emb)   # MLP layers  → (B, H')
            logits = self.model.fc(rep)      # classifier  → (B, N)
            _, pred = logits.max(-1)

            acc = float((pred == labels).long().sum()) / labels.size(0)
            avg_acc.update(acc, labels.size(0))

            if training:
                loss = self.criterion(logits, labels)
                avg_loss.update(loss.item(), 1)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)

        return avg_loss.avg if training else None

    def eval_model(
        self,
        eval_loader: DataLoader,
    ) -> tuple[dict[str, float], list[int], list[int]]:
        """Evaluate on *eval_loader* using pre-computed embeddings."""
        self.eval()
        pred_result: list[int] = []
        all_labels:  list[int] = []
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for emb, labels in tqdm(eval_loader):
                emb    = emb.to(device)
                labels = labels.to(device)
                rep    = self.model.model(emb)
                logits = self.model.fc(rep)
                _, pred = logits.max(-1)
                pred_result.extend(pred.tolist())
                all_labels.extend(labels.tolist())

        result = {
            "acc":     accuracy_score(all_labels, pred_result),
            "micro_p": precision_score(all_labels, pred_result, average="micro", zero_division=0),
            "micro_r": recall_score(all_labels, pred_result, average="micro", zero_division=0),
            "micro_f1": f1_score(all_labels, pred_result, average="micro", zero_division=0),
            "macro_p": precision_score(all_labels, pred_result, average="macro", zero_division=0),
            "macro_r": recall_score(all_labels, pred_result, average="macro", zero_division=0),
            "macro_f1": f1_score(all_labels, pred_result, average="macro", zero_division=0),
        }
        logger.info("Eval result: %s", result)
        return result, pred_result, all_labels