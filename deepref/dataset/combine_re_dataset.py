import ast
import logging
import torch

from deepref.dataset.re_dataset import REDataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

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
    


