import os
import json
import argparse
import pandas as pd
from opennre import config
from pathlib import Path
from opennre.framework.train import Training

class AblationStudies():
    def __init__(self, dataset, model, embeddings=[], best_hparams=True):
        self.dataset = dataset
        self.model = model
        self.embeddings = embeddings
        self.csv_path = f'opennre/ablation/{self.dataset}_{self.model}_ablation_studies.csv'
        self.ablation = {
            'preprocessing': [], 
            'embeddings': [], 
            'acc': [], 
            'micro_p':[], 
            'micro_r': [], 
            'micro_f1': [], 
            'macro_f1': []
        }
        self.embeddings_combination = self.embed_combinations(len(config.TYPE_EMBEDDINGS))
        self.exp = 0
        
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            ablation = df.to_dict('split')
            for data in ablation["data"]:
                self.ablation['preprocessing'].append(data[0])
                self.ablation['embeddings'].append(data[1])
                self.ablation['acc'].append(data[2])
                self.ablation['micro_p'].append(data[3])
                self.ablation['micro_r'].append(data[4])
                self.ablation['micro_f1'].append(data[5])
                self.ablation['macro_f1'].append(data[6])
            self.exp = len(self.ablation['preprocessing'])
            print(len(self.ablation["preprocessing"]))
        
        if not os.path.exists(config.BEST_HPARAMS_FILE_PATH.format(dataset)) or not best_hparams:
            dict_params = config.HPARAMS
            json_object = json.dumps(dict_params, indent=4)
            with open(config.BEST_HPARAMS_FILE_PATH.format(dataset), 'w') as f:
                f.write(json_object)
        self.best_hparams = {}
        with open(config.BEST_HPARAMS_FILE_PATH.format(dataset), 'r') as f:
            self.best_hparams = json.load(f)
            
    def execute_ablation(self):
        parameters = self.best_hparams
        parameters["dataset"] = self.dataset

        index = 0
        embed_indexes = [config.TYPE_EMBEDDINGS.index(embed) for embed in self.embeddings]
        for preprocessing in config.PREPROCESSING_COMBINATION:
            for embed in self.embeddings_combination:
                has_embed = sum([embed[idx] for idx in embed_indexes]) == len(embed_indexes)
                if not has_embed:
                    continue
                if index > self.exp - 1:
                
                    parameters["model"] = self.model
                    parameters["pos_tags_embed"] = embed[config.TYPE_EMBEDDINGS.index('pos_tags')]
                    parameters["deps_embed"] = embed[config.TYPE_EMBEDDINGS.index('deps')]
                    parameters["sk_embed"] = embed[config.TYPE_EMBEDDINGS.index('sk')]
                    parameters["position_embed"] = embed[config.TYPE_EMBEDDINGS.index('position')]
                    parameters["sdp_embed"] = 0
                    parameters["preprocessing"] = preprocessing
                    
                    train = Training(self.dataset, parameters)
                    
                    result = train.train()
                    acc = result["acc"]
                    micro_p = result["micro_p"]
                    micro_r = result["micro_r"]
                    micro_f1 = result["micro_f1"]
                    macro_f1 = result["macro_f1"]
                    
                    embeddings = ''
                    for i in range(len(config.TYPE_EMBEDDINGS)):
                        embeddings += ' ' + config.TYPE_EMBEDDINGS[i] * embed[i]
                    embeddings = embeddings.strip()
                    self.ablation["acc"].append(acc)
                    self.ablation["micro_p"].append(micro_p)
                    self.ablation["micro_r"].append(micro_r)
                    self.ablation["micro_f1"].append(micro_f1)
                    self.ablation["macro_f1"].append(macro_f1)
                    self.ablation["embeddings"].append(embeddings)
                    self.ablation["preprocessing"].append(preprocessing)
                    
                    self.save_ablation()
                
                index += 1
                
        return self.ablation
        
    def save_ablation(self):
        df = pd.DataFrame.from_dict(self.ablation)
        filepath = Path(f'opennre/ablation/{self.dataset}_{self.model}_ablation_studies.csv')
        df.to_csv(filepath, index=False)
        
    def embed_combinations(self, number_of_combinations):
        combinations = []
        sum_binary = bin(0)
        for _ in range(2**number_of_combinations):
            list_comb = [int(i) for i in list(sum_binary[2:])]
            list_comb = ((number_of_combinations - len(list_comb)) * [0]) + list_comb
            sum_binary = bin(int(sum_binary, 2) + int("1", 2))
            
            if len(list_comb) == number_of_combinations:
                combinations.append(list_comb)
            
        return combinations
            
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', default="semeval2010", choices=config.DATASETS, 
                help='Dataset')
    parser.add_argument('-m','--model', default="bert_entity", choices=config.MODELS, 
                help='Models')
    parser.add_argument('-e','--embeddings', nargs="+", default=[], choices=config.TYPE_EMBEDDINGS, 
                help='Embeddings')
    parser.add_argument('-p','--preprocessing', nargs="+", default=[],
                help='Preprocessing')
    parser.add_argument('--best_hyperparameters', action='store_true', default=True,
        help='Run with best hyperparameters (True) or default (False)')
    args = parser.parse_args()
    
    ablation = AblationStudies(args.dataset, args.model, args.embeddings, args.best_hyperparameters)
    ablation.execute_ablation()
