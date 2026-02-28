import ast
import logging
from pathlib import Path

import pandas as pd

import sklearn
import torch
from torch.utils.data import DataLoader, Dataset

class REDataset(Dataset):
    def __init__(self, csv_path, tokenizer, preprocessors_list=[]) -> None:
        self.df = self.get_dataframe(csv_path)
        self.tokenizer = tokenizer
        self.rel2id = self.get_labels_dict()
        self.preprocessors_list = preprocessors_list
        self.max_length = self.get_max_length()

    def __len__(self):
        if self.df is None:
            return 0
        return len(self.df)

    def __getitem__(self, index):
        if self.df is None:
            return None
        sentence = self.df.iloc[index]["original_sentence"]
        e1 = self.df.iloc[index]["e1"]
        e2 = self.df.iloc[index]["e2"]
        sentence = self.format_sentence(sentence, e1, e2)

        label = self.df.iloc[index]["relation_type"]
        for preprocessor in self.preprocessors_list:
            sentence = preprocessor(sentence)
        seq = self.tokenizer(sentence,
                                  max_length=self.max_length, 
                                  padding="max_length", 
                                  truncation=True, 
                                  return_tensors="pt")
        item = {k: torch.tensor(v).squeeze(0) for k, v in seq.items()}
        item["labels"] = torch.tensor(self.rel2id[label], dtype=torch.long)
        return item
    
    def format_sentence(self, sentence, e1, e2):
        marks = ["<e1>", "</e1>", "<e2>", "</e2>"]
        sentennce_splitted = sentence.split()
        e1_position = ast.literal_eval(e1)["position"]
        e2_position = ast.literal_eval(e2)["position"]
            
        for pos in e2_position[::-1]:
            sentennce_splitted.insert(pos, marks.pop())

        for pos in e1_position[::-1]:
            sentennce_splitted.insert(pos, marks.pop())
        
        return " ".join(sentennce_splitted)

    def get_dataframe(self, csv_path: str) -> pd.DataFrame | None:
        path = Path(csv_path)
        if not path.suffix:
            path = path.with_suffix('.csv')
        if not path.is_file():
            return None
        return pd.read_csv(path, sep="\t")
            
    def get_labels_dict(self):
        if self.df is None:
            return {}
        relation_types = list(self.df["relation_type"].unique())
        relation_types.sort()
        rel2id = {relation_types[i]: i for i in range(len(relation_types))}
        return rel2id
    
    def get_max_length(self):
        if self.df is None:
            return 0
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
        total = len(self.df) if self.df is not None else 0
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
                golden = self.df.iloc[i]['relation_type'] if self.df is not None else "NA"
                goldens.append(golden)
            else:
                golden = self.rel2id[self.df.iloc[i]['relation_type']] if self.df is not None else -1
                goldens.append(golden)
            if golden == pred_result[i]:
                correct += 1
                if golden != neg and golden in correct_positive:
                    correct_positive[golden] += 1
            if golden != neg and golden in gold_positive:
                gold_positive[golden] +=1
            if pred_result[i] != neg and golden in pred_positive:
                pred_positive[golden] += 1
        acc = float(correct) / float(total)

        #Micro
        try:
            micro_p = self.calculate_micro_precision(correct_positive, pred_positive)
            macro_p = self.calculate_macro_precision(correct_positive, pred_positive)
        except:
            micro_p = 0
            macro_p = 0
        try:
            micro_r = self.calculate_micro_recall(correct_positive, gold_positive)
            macro_r = self.calculate_macro_recall(correct_positive, gold_positive)
        except:
            micro_r = 0
            macro_r = 0
        try:
            micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
            macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r)
        except:
            micro_f1 = 0
            macro_f1 = 0
            
        confusion_matrix = sklearn.metrics.confusion_matrix(goldens, pred_result)

        result = {'acc': acc, 
                  'micro_p': micro_p, 
                  'micro_r': micro_r, 
                  'micro_f1': micro_f1, 
                  'macro_f1': macro_f1, 
                  'cm': confusion_matrix}
        logging.info('Evaluation result: \n {}.'.format(result))
        return result
    
    def calculate_micro_precision(self, correct_positive, pred_positive):
        return float(sum(correct_positive.values())) / float(sum(pred_positive.values()))
    
    def calculate_macro_precision(self, correct_positive, pred_positive):
        return sum([cp/list(pred_positive.values())[i] for i, cp in enumerate(correct_positive.values())]) / len(self.rel2id)
    
    def calculate_micro_recall(self, correct_positive, gold_positive):
        return float(sum(correct_positive.values())) / float(sum(gold_positive.values()))
    
    def calculate_macro_recall(self, correct_positive, gold_positive):
        return sum([cp/list(gold_positive.values())[i] for i, cp in enumerate(correct_positive.values())]) / len(self.rel2id)
    
def RELoader(dataset: REDataset, batch_size, 
        shuffle, num_workers=0):
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=num_workers)
    return data_loader