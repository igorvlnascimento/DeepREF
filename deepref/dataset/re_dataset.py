import logging
from pathlib import Path

import pandas as pd

import sklearn
import torch
from torch.utils.data import DataLoader, Dataset

class REDataset(Dataset):
    def __init__(self, csv_path, tokenizer, dataset_split="train", preprocessor=None) -> None:
        self.df = self.get_dataframe(csv_path, dataset_split=dataset_split)
        self.tokenizer = tokenizer
        self.rel2id = self.get_labels_dict()
        self.preprocessor = preprocessor
        self.max_length = self.get_max_length()

    def __len__(self):
        if self.df is None:
            return 0
        return len(self.df)

    def __getitem__(self, index):
        if self.df is None:
            return None
        sentence = self.df.iloc[index]["original_sentence"]
        label = self.df.iloc[index]["relation_type"]
        if self.preprocessor is not None:
            sentence = self.preprocessor(sentence)
        seq = list(self.tokenizer(sentence,
                                  max_length=self.max_length, 
                                  padding="max_length", 
                                  truncation=True, 
                                  return_tensors="pt").values())
        return [self.rel2id[label]] + seq
    
    def collate_fn(data):
        data = list(zip(*data))
        labels = data[0]
        seqs = data[1:]
        batch_labels = torch.tensor(labels).long() # (B)
        batch_seqs = []
        for seq in seqs:
            #print(seq)
            batch_seqs.append(torch.cat(seq, 0)) # (B, L)
        return [batch_labels] + batch_seqs

    def get_dataframe(self, csv_dir: str, dataset_split: str) -> pd.DataFrame | None:
        base = Path(csv_dir)
        csv_files = list(base.rglob("*.csv"))

        for csv_file in csv_files:
            if dataset_split in csv_file.name:
                return pd.read_csv(csv_file, sep="\t")
            
    def get_labels_dict(self):
        if self.df is None:
            return {}
        relation_types = list(self.df["relation_type"].unique())
        relation_types.sort()
        rel2id = {relation_types[i]: i for i in range(len(relation_types))}
        return rel2id
    
    def get_max_length(self):
        lengths = self.df["original_sentence"].apply(
            lambda x: len(self.tokenizer(x, add_special_tokens=True)["input_ids"])
        )
        return lengths.max()
    
    def eval(self, pred_result, use_name=False):
        """
        Args:
            pred_result: a list of predicted label (id)
                Make sure that the `shuffle` param is set to `False` when getting the loader.
            use_name: if True, `pred_result` contains predicted relation names instead of ids
        Return:
            {'acc': xx}
        """
        correct = 0
        goldens = []
        total = len(self.df)
        correct_positive = 0
        pred_positive = 0
        gold_positive = 0
        correct_positive = {k:0 for k in range(len(self.rel2id))}
        pred_positive = {k:0 for k in range(len(self.rel2id))}
        gold_positive = {k:0 for k in range(len(self.rel2id))}
        neg = -1
        for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'none', 'None', 'int']:
            if name in self.rel2id:
                if use_name:
                    neg = name
                else:
                    neg = self.rel2id[name]
                break
        for i in range(total):
            if use_name:
                golden = self.df.iloc[i]['relation_type']
                goldens.append(golden)
            else:
                golden = self.rel2id[self.df.iloc[i]['relation_type']]
                goldens.append(golden)
            if golden == pred_result[i]:
                correct += 1
                if golden != neg:
                    correct_positive[golden] += 1
                    #correct_positive += 1
            if golden != neg:
                gold_positive[golden] +=1
            if pred_result[i] != neg:
                pred_positive[golden] += 1
        acc = float(correct) / float(total)

        #Micro
        try:
            micro_p = float(sum(correct_positive.values())) / float(sum(pred_positive.values()))
            macro_p = sum([cp/list(pred_positive.values())[i] for i, cp in enumerate(correct_positive.values())]) / len(self.rel2id)
        except:
            micro_p = 0
            macro_p = 0
        try:
            micro_r = float(sum(correct_positive.values())) / float(sum(gold_positive.values()))
            macro_r = sum([cp/list(gold_positive.values())[i] for i, cp in enumerate(correct_positive.values())]) / len(self.rel2id)
        except:
            micro_r = 0
            macro_r = 0
        try:
            micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
            macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r)
        except:
            micro_f1 = 0
            macro_f1 = 0
            
        #micro_f1 = sklearn.metrics.f1_score(goldens, pred_result, labels=list(range(len(self.rel2id))), average='micro')
        #macro_f1 = sklearn.metrics.f1_score(goldens, pred_result, labels=list(range(len(self.rel2id))), average='macro')
        confusion_matrix = sklearn.metrics.confusion_matrix(goldens, pred_result)

        result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1, 'macro_f1': macro_f1, 'cm': confusion_matrix}
        logging.info('Evaluation result: \n {}.'.format(result))
        return result
    
def RELoader(dataset: REDataset, batch_size, 
        shuffle, num_workers=0, collate_fn=REDataset.collate_fn):
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=num_workers,
        collate_fn=collate_fn)
    return data_loader