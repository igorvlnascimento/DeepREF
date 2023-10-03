import os
import json
import argparse
import random
import pandas as pd
import numpy as np
from deepref import config
from pathlib import Path
from deepref.framework.train import Training

class AblationStudies():
    def __init__(self, dataset):
        self.dataset = dataset
        
        if not os.path.exists(config.HPARAMS_FILE_PATH.format(dataset)):
            dict_params = config.HPARAMS
            json_object = json.dumps(dict_params, indent=4)
            with open(config.HPARAMS_FILE_PATH.format(dataset), 'w') as f:
                f.write(json_object)
        self.hparams = {}
        with open(config.HPARAMS_FILE_PATH.format(dataset), 'r') as f:
            self.hparams = json.load(f)
            
        self.model = self.hparams["model"]
        self.csv_path = f'deepref/ablation/{self.dataset}_{self.model}_ablation_studies.csv'
        
        self.ablation = {
            'model': [],
            'pretrain': [],
            'preprocessing': [], 
            'embeddings': [], 
            'acc': [], 
            'micro_p':[], 
            'micro_r': [], 
            'micro_f1': [], 
            'macro_f1': [],
            'trial': []
        }
        self.embeddings_combination = self.embed_combinations(len(config.TYPE_EMBEDDINGS))
        self.exp = 0
        
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            ablation = df.to_dict('split')
            for data in ablation["data"]:
                self.ablation['model'].append(data[0])
                self.ablation['pretrain'].append(data[1])
                self.ablation['preprocessing'].append(data[2])
                self.ablation['embeddings'].append(data[3])
                self.ablation['acc'].append(data[4])
                self.ablation['micro_p'].append(data[5])
                self.ablation['micro_r'].append(data[6])
                self.ablation['micro_f1'].append(data[7])
                self.ablation['macro_f1'].append(data[8])
            self.exp = len(self.ablation['preprocessing'])
            print(len(self.ablation["preprocessing"]))
            
    def execute_ablation(self):
        parameters = self.hparams

        index = 0
        seeds = self.get_seeds()
        #print(config.TYPE_EMBEDDINGS_COMBINATION)
        #embed_indexes = [config.TYPE_EMBEDDINGS.index(embed) for embed in config.TYPE_EMBEDDINGS_COMBINATION]
        for model in config.MODELS[-3:]:
            for pretrain in config.PRETRAIN_WEIGHTS:
                for i, preprocessing in enumerate(config.PREPROCESSING_COMBINATION):
                    for j, embed in enumerate(self.embeddings_combination):
                        #has_embed = sum([embed[idx] for idx in embed_indexes]) == len(embed_indexes)
                        if (i+1)*(j+1) < self.exp:
                            continue

                        acc_list = []
                        micro_f1_list = []
                        macro_f1_list = []
                        for k in range(3):
                            parameters["model"] = model
                            parameters["pretrain"] = pretrain
                            parameters["pos_tags_embed"] = embed[config.TYPE_EMBEDDINGS.index('pos_tags')]
                            parameters["deps_embed"] = embed[config.TYPE_EMBEDDINGS.index('deps')]
                            parameters["position_embed"] = embed[config.TYPE_EMBEDDINGS.index('position')]
                            parameters["preprocessing"] = preprocessing
                            
                            train = Training(self.dataset, parameters, seed=seeds[k])
                            
                            result = train.train()
                            acc = result["acc"]
                            micro_p = result["micro_p"]
                            micro_r = result["micro_r"]
                            micro_f1 = result["micro_f1"]
                            macro_f1 = result["macro_f1"]

                            acc_list.append(acc)
                            micro_f1_list.append(micro_f1)
                            macro_f1_list.append(macro_f1)
                            
                            embeddings = ''
                            for i in range(len(config.TYPE_EMBEDDINGS)):
                                embeddings += ' ' + config.TYPE_EMBEDDINGS[i] * embed[i]
                            embeddings = embeddings.strip()
                            self.ablation["acc"].append(acc)
                            self.ablation["micro_p"].append(micro_p)
                            self.ablation["micro_r"].append(micro_r)
                            self.ablation["micro_f1"].append(micro_f1)
                            self.ablation["macro_f1"].append(macro_f1)
                            self.ablation["model"].append(model)
                            self.ablation["pretrain"].append(pretrain)
                            self.ablation["embeddings"].append(embeddings)
                            self.ablation["preprocessing"].append(preprocessing)
                            self.ablation["trial"].append(k+1)
                            
                            self.save_ablation()
                        
                        self.ablation["acc"].append(np.mean(acc_list))
                        self.ablation["micro_p"].append(0)
                        self.ablation["micro_r"].append(0)
                        self.ablation["micro_f1"].append(np.mean(micro_f1_list))
                        self.ablation["macro_f1"].append(np.mean(macro_f1_list))
                        self.ablation["model"].append(model)
                        self.ablation["pretrain"].append(pretrain)
                        self.ablation["embeddings"].append('mean')
                        self.ablation["preprocessing"].append('')
                        self.ablation["trial"].append(0)

                        self.ablation["acc"].append(np.std(acc_list))
                        self.ablation["micro_p"].append(0)
                        self.ablation["micro_r"].append(0)
                        self.ablation["micro_f1"].append(np.std(micro_f1_list))
                        self.ablation["macro_f1"].append(np.std(macro_f1_list))
                        self.ablation["model"].append(model)
                        self.ablation["pretrain"].append(pretrain)
                        self.ablation["embeddings"].append('std')
                        self.ablation["preprocessing"].append('')
                        self.ablation["trial"].append(0)
                        
                        index += 1
                        
        return self.ablation
        
    def save_ablation(self):
        df = pd.DataFrame.from_dict(self.ablation)
        filepath = Path(f'deepref/ablation/{self.dataset}_{self.model}_ablation_studies.csv')
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
            
    def get_seeds(self):
        seeds = []
        seeds_path = 'deepref/seeds.txt'
        if os.path.exists(seeds_path):
            seeds = self.read_seeds()
        else:
            seeds = self.generate_seeds()
            self.write_seeds(seeds)
        return seeds

    def generate_seeds(self):
        seeds = []
        for _ in range(3):
            seeds.append(random.randint(10**6, 10**7-1))
        return seeds
    
    def write_seeds(self, seeds):
        filepath = Path(f'deepref/seeds.txt')
        with open(filepath, 'w') as file:
            for seed in seeds:
                file.write(str(seed) + '\n')
    
    def read_seeds(self):
        filepath = Path(f'deepref/seeds.txt')
        with open(filepath, 'r') as file:
            seeds = file.readlines()
            seeds = [int(seed) for seed in seeds]
        return seeds
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', default="semeval2010", choices=config.DATASETS, 
                help='Dataset')
    args = parser.parse_args()
    
    ablation = AblationStudies(args.dataset)
    ablation.execute_ablation()
