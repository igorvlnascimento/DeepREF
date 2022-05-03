import argparse
import json
import os

import pandas as pd

from opennre import config

from pathlib import Path

from opennre.framework.train import Training

EMBEDDINGS_COMBINATION = [[0,0],[0,1],[1,0],[1,1]]

class AblationStudies():
    def __init__(self, dataset, model, best_hparams):
        self.dataset = dataset
        self.model = model
        self.csv_path = f'opennre/ablation/{self.dataset}_{self.model}_ablation_studies.csv'
        self.ablation = {'preprocessing': [], 'embeddings': [], 'micro_f1': [], 'macro_f1': []}
        
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            self.ablation = df.to_dict('split')
            print(self.ablation)
        
        if not os.path.exists(config.BEST_HPARAMS_FILE_PATH.format(dataset)) or not best_hparams:
            dict = config.HPARAMS
            json_object = json.dumps(dict, indent=4)
            with open(config.BEST_HPARAMS_FILE_PATH.format(dataset), 'w') as f:
                f.write(json_object)
        self.best_hparams = {}
        with open(config.BEST_HPARAMS_FILE_PATH.format(dataset), 'r') as f:
            self.best_hparams = json.load(f)
            
    def execute_ablation(self, having_preprocessing=None):
        parameters = self.best_hparams
        parameters["dataset"] = self.dataset
        
        for preprocessing in config.PREPROCESSING_COMBINATION:
            if having_preprocessing is not None:
                if having_preprocessing not in preprocessing:
                    continue
                    
            #for embed in EMBEDDINGS_COMBINATION:
            embed = [0,0]
            parameters["pos_embed"] = embed[0]
            parameters["deps_embed"] = embed[1]
            parameters["preprocessing"] = config.PREPROCESSING_COMBINATION.index(preprocessing)
            
            train = Training(self.dataset, parameters)
            
            result = train.train()
            micro_f1 = result["micro_f1"]
            macro_f1 = result["macro_f1"]
            
            self.ablation["preprocessing"].append(preprocessing)
            embeddings = ("pos" * embed[0] + '_' * (sum(embed) - 1) + "deps" * embed[1])
            self.ablation["embeddings"].append(embeddings)
            self.ablation["macro_f1"].append(macro_f1)
            self.ablation["micro_f1"].append(micro_f1)
            
            print(self.ablation)
            self.save_ablation()
            
        return self.ablation
        
    def save_ablation(self):
        df = pd.DataFrame.from_dict(self.ablation)
        filepath = Path(f'opennre/ablation/{self.dataset}_{self.model}_ablation_studies.csv')
        #filepath.parent.mkdir(parents=True, exist_ok=True) 
        df.to_csv(filepath, index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', default="semeval2010", choices=config.DATASETS, 
                help='Dataset')
    parser.add_argument('-m','--model', default="bert_cls", choices=config.MODELS, 
                help='Models')
    parser.add_argument('--best_hyperparameters', action='store_true', 
        help='Run with best hyperparameters (True) or default (False)')
    args = parser.parse_args()
    
    ablation = AblationStudies(args.dataset, args.model, args.best_hyperparameters)
    ablation.execute_ablation()
