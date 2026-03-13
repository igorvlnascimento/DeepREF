# ---------------------------------------------------------------------------
# Sklearn-based classifier wrapper
# ---------------------------------------------------------------------------

from lightgbm import LGBMClassifier
import torch
from torch import nn
from xgboost import XGBClassifier


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
        sentence_encoder: nn.Module | None,
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
            return XGBClassifier(
                objective="multi:softprob",
                num_class=num_class,
                eval_metric="mlogloss",
                n_estimators=100,
                use_label_encoder=False,
            )
        if model_type == "lightgbm":
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

    def forward_from_emb(self, emb: torch.Tensor) -> torch.Tensor:
        """Classify pre-computed embeddings — compatible with HybridRETrainer.

        Args:
            emb: float tensor of shape ``(B, H)``.

        Returns:
            Class-probability tensor of shape ``(B, C)``.
        """
        import numpy as np

        X = emb.cpu().numpy()
        proba = self._clf.predict_proba(X)
        return torch.from_numpy(np.array(proba, dtype=np.float32))

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