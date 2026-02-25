from pathlib import Path

import pandas as pd

import torch
from torch.utils.data import Dataset

class REDataset(Dataset):
    def __init__(self, csv_path, dataset_split="train", preprocessor=None) -> None:
        self.df = self.get_dataframe(csv_path, dataset_split=dataset_split)
        self.preprocessor = preprocessor

    def __len__(self):
        if self.df is None:
            return 0
        return len(self.df)

    def __getitem__(self, index):
        if self.df is None:
            return None
        return self.df.iloc[index]

    def get_dataframe(self, csv_dir: str, dataset_split: str) -> pd.DataFrame | None:
        base = Path(csv_dir)
        csv_files = list(base.rglob("*.csv"))

        for csv_file in csv_files:
            if dataset_split in csv_file.name:
                return pd.read_csv(csv_file, sep="\t")