from typing import Any

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from torch import nn


class PCAEncoder(nn.Module):
    def __init__(self, n_components: float = 0.95) -> None:
        super().__init__()
        self.preprocessing_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("l2_norm", Normalizer(norm="l2"))
        ])

    def forward(self, x: Any) -> Any:
        return self.preprocessing_pipeline.fit_transform(x)