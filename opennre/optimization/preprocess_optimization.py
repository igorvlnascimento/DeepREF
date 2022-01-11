import itertools
import argparse
import json
import os

from train import Training

CONFIG_FILE_PATH = "opennre/optimization/config_params.json"
BEST_HPARAMS_FILE_PATH = "opennre/optimization/best_hparams_{}.json"

class PreprocessOptimization():
    def __init__(self, dataset, metric):
        self.dataset = dataset
        self.metric = metric
        self.data = json.load(open(CONFIG_FILE_PATH))
        self.preprocessing = self.data["preprocessing"]
        self.preprocess_combination = self.combine_preprocessing(self.preprocessing)
        
        synt_embeddings = [0,0,0]
        batch_size = 32
        if dataset == 'semeval2010':
            batch_size = 16
        #     synt_embeddings = [1,1,1]
        # elif dataset == 'ddi':
        #     synt_embeddings = [1,0,1]
        # elif dataset == 'semeval20181-1':
        #     synt_embeddings = [1,1,0]
        # elif dataset == 'semeval20181-2':
        #     synt_embeddings = [1,0,1]
        
        if not os.path.exists(BEST_HPARAMS_FILE_PATH.format(dataset)):
            dict = {
                "{}".format(self.metric): 0,
                "batch_size": batch_size,
                "preprocessing": 0,
                "lr": 2e-5,
                "synt_embeddings": synt_embeddings,
                "max_length": 128,
                "max_epoch": 3
            }
            json_object = json.dumps(dict, indent=4)
            with open(BEST_HPARAMS_FILE_PATH.format(dataset), 'w') as f:
                f.write(json_object)
        self.best_hparams = {}
        with open(BEST_HPARAMS_FILE_PATH.format(dataset), 'r') as f:
            self.best_hparams = json.load(f)

    def combine_preprocessing(self, preprocessing):
            combinations = []
            for i in range(len(preprocessing)):
                combinations.extend(itertools.combinations(preprocessing, i))
                
            for j, comb in enumerate(combinations):
                if 'eb' in comb and 'nb' in comb:
                    comb = list(comb)
                    comb.remove('eb')
                    combinations[j] = comb
                else:
                    combinations[j] = list(comb)
            
            final_combinations = [comb for n, comb in enumerate(combinations) if comb not in combinations[:n]]
            print(final_combinations)
            return final_combinations
        
    def preprocessing_training(self):
        model = 'bert',#self.study_model.best_params["model"]
        pretrain_bert = 'bert-base-uncased' if self.dataset == 'semeval2010' else 'allenai/scibert_scivocab_uncased'#individual.suggest_categorical("pretrain_bert", self.data["pretrain_bert"])
        synt_embeddings = self.best_hparams["synt_embeddings"]

        batch_size =  self.best_hparams["batch_size"]
        lr =  self.best_hparams["lr"]
        max_length = self.best_hparams["max_length"]
        max_epoch = self.best_hparams["max_epoch"]
        
        preprocessing_type, preprocessing_value = 0, 0
        
        for i in range(len(self.preprocess_combination)):
        
            parameters = {
                "dataset": self.dataset,
                "model": model,
                "metric": self.metric,
                "preprocessing": self.preprocess_combination[i],
                "embedding": pretrain_bert,
                "synt_embeddings": synt_embeddings,
                "batch_size": batch_size,#batch_size_bert if model == 'bert' else batch_size,
                "lr": lr,
                "weight_decay": None,#weight_decay,
                "max_length": max_length,
                "max_epoch": max_epoch,#max_epoch_bert if model == 'bert' else max_epoch,
                "pooler": None,
                "opt": None,
                "mask_entity": None,
                "hidden_size": None,
                "position_size": None,
                "dropout": None,
            }
            
            train = Training(parameters)
            
            new_value = train.train()
            
            if new_value > preprocessing_value:
                preprocessing_type, preprocessing_value = i, new_value 
            
        return preprocessing_type, preprocessing_value
    
    

    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', default="semeval2010", choices=["semeval2010", "semeval2018", "semeval20181-1", "semeval20181-2", "ddi"], 
                help='Dataset')
    parser.add_argument('-m','--metric', default="micro_f1", choices=["micro_f1", "macro_f1", "acc"], 
                help='Metric to optimize')
    args = parser.parse_args()
    dataset = args.dataset
    metric = args.metric
    prep = PreprocessOptimization(dataset, metric)
    preprocessing, new_value = prep.preprocessing_training()
    print("Type:", prep.preprocess_combination[preprocessing], "Value:", new_value)
    
    best_hparams = {}
    with open(BEST_HPARAMS_FILE_PATH.format(dataset), 'r') as f:
        best_hparams = json.load(f)
        
    json_value = float(best_hparams["{}".format(metric)]) if best_hparams["{}".format(metric)] else 0
    
    if new_value > json_value:
        best_hparams["preprocessing"] = preprocessing
        #best_hparams["synt_embeddings"] = synt_embeddings
        best_hparams["{}".format(metric)] = new_value
        json_object = json.dumps(best_hparams, indent=4)
        
        with open(BEST_HPARAMS_FILE_PATH.format(dataset), 'w') as out_f:
            out_f.write(json_object)
    #print(preprocessing[30])