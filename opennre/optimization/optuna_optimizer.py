import os
import json
import argparse
import itertools

import optuna

from train import Training

CONFIG_FILE_PATH = "opennre/optimization/config_params.json"
BEST_HPARAMS_FILE_PATH = "opennre/optimization/best_hparams_{}.json"

class Optimizer():
    def __init__(self, dataset, metric):
        self.dataset = dataset
        self.metric = metric
        self.data = json.load(open(CONFIG_FILE_PATH))
        if not os.path.exists(BEST_HPARAMS_FILE_PATH):
            dict = {
                "{}".format(self.metric): 0,
                "batch_size": 16,
                "preprocessing": "original",
                "lr": 1e-5,
                "max_length": 128,
                "max_epoch": 3,
                "synt_embeddings": [0,0]
            }
            json_object = json.dumps(dict, indent=4)
            with open(BEST_HPARAMS_FILE_PATH.format(dataset), 'w') as f:
                f.write(json_object)
        self.best_hparams = {}
        with open(BEST_HPARAMS_FILE_PATH.format(dataset), 'r') as f:
            self.best_hparams = json.load(f)
        
        self.study_model = optuna.create_study()
        self.study_params = optuna.create_study()
    
        self.final_combinations = []
        
        self.preprocessing = self.data["preprocessing"]
        self.preprocessing = self.combine_preprocessing(self.preprocessing)
        
        self.synt_embeddings = [[0,0], [0,1], [1,0], [1,1]]
        
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
        
        self.final_combinations = [comb for n, comb in enumerate(combinations) if comb not in combinations[:n]]
        print(self.final_combinations)
        return self.final_combinations

    def evaluate_model(self, individual):
        
        model = 'bert',#individual.suggest_categorical("model", self.data["model"])
        #preprocessing =  individual.suggest_int("preprocessing", 0, len(self.preprocessing)-1)
        synt_embeddings = individual.suggest_int("synt_embeddings", 0, len(self.synt_embeddings)-1)
        pretrain_bert = 'deepset/sentence_bert' if self.dataset == 'semeval2010' else 'allenai/scibert_scivocab_uncased'#individual.suggest_categorical("pretrain_bert", self.data["pretrain_bert"])
        
        #batch_size_bert =  individual.suggest_int("batch_size_bert", 32, 128, log=True)
        batch_size =  individual.suggest_int("batch_size", 32, 128, log=True)
        lr =  individual.suggest_float("lr", 1e-6, 1e-1, log=True)
        #weight_decay =  individual.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        max_length =  individual.suggest_int("max_length", 32, 256, log=True)
        #max_epoch_bert =  individual.suggest_int("max_epoch_bert", 2, 8, log=True)
        max_epoch = individual.suggest_int("max_epoch", 2, 8)
    
        parameters = {
            "dataset": self.dataset,
            "model": model,
            "metric": self.metric,
            "preprocessing": [],#self.preprocessing[preprocessing],
            "embedding": pretrain_bert,# if model == "bert" else embedding,
            "synt_embeddings": self.synt_embeddings[synt_embeddings],
            "pooler": None,
            "opt": None,
            "batch_size": batch_size,#_bert if model == "bert" else batch_size,
            "lr": lr,
            "weight_decay": None,#weight_decay,
            "max_length": max_length,
            "max_epoch": max_epoch,#_bert if model == 'bert' else max_epoch,
            "mask_entity": None,
            "hidden_size": None,
            "position_size": None,
            "dropout": None,
        }
        
        print("parameters:",parameters)
        
        train = Training(parameters)
        
        return -(train.train())
        
    def evaluate_preprocessing(self, individual):
    
        preprocessing =  individual.suggest_int("preprocessing", 0, len(self.preprocessing)-1)
        model = 'bert',#self.study_model.best_params["model"]
        pretrain_bert = 'deepset/sentence_bert' if self.dataset == 'semeval2010' else 'allenai/scibert_scivocab_uncased'#individual.suggest_categorical("pretrain_bert", self.data["pretrain_bert"])
        synt_embeddings = self.best_hparams["synt_embeddings"]

        batch_size =  self.best_hparams["batch_size"]
        lr =  self.best_hparams["lr"]
        max_length =  self.best_hparams["max_length"]
        max_epoch = self.best_hparams["max_epoch"]
        
        parameters = {
            "dataset": self.dataset,
            "model": model,
            "metric": self.metric,
            "preprocessing": self.preprocessing[preprocessing],
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
        
        return -(train.train())
    
    def evaluate_hyperparameters(self, individual):
        
        #preprocessing = self.preprocessing[self.study.best_params["preprocessing"]]
        model = 'bert',#self.study_model.best_params["model"]
        #embedding = self.study_model.best_params["pretrain_bert"] if model == 'bert' else self.study_model.best_params["embedding"]
        pretrain_bert = 'deepset/sentence_bert' if self.dataset == 'semeval2010' else 'allenai/scibert_scivocab_uncased'#individual.suggest_categorical("pretrain_bert", self.data["pretrain_bert"])
        
        #preprocessing =  individual.suggest_int("preprocessing", 0, len(self.preprocessing)-1)
        #batch_size_bert =  individual.suggest_int("batch_size_bert", 32, 128, log=True)
        batch_size =  individual.suggest_int("batch_size", 32, 256, log=True)
        lr =  individual.suggest_float("lr", 1e-5, 1e-1, log=True)
        #weight_decay =  individual.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        max_length =  individual.suggest_int("max_length", 8, 256, log=True)
        #max_epoch_bert =  individual.suggest_int("max_epoch_bert", 2, 8, log=True)
        max_epoch = individual.suggest_int("max_epoch", 2, 8)
    
        parameters = {
            "dataset": self.dataset,
            "model": model,
            "metric": self.metric,
            "preprocessing": None,#self.preprocessing[preprocessing],
            "embedding": pretrain_bert,
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
        
        return -(train.train())

    def optimize_model(self):
        self.study_model.optimize(self.evaluate_model, n_trials=50)
    
        params = self.study_model.best_params
        
        return params
    
    def optimize_preprocessing(self):
        self.study_model.optimize(self.evaluate_preprocessing, n_trials=50)
    
        params = self.study_model.best_params
        
        return params
        
    # def optimize_hyperparameters(self):
    #     self.study_params.optimize(self.evaluate_hyperparameters, n_trials=50)
    
    #     params = self.study_params.best_params
        
    #     return params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', default="semeval2010", choices=["semeval2010", "semeval2018", "semeval20181-1", "semeval20181-2", "ddi"], 
                help='Dataset')
    parser.add_argument('-m','--metric', default="micro_f1", choices=["micro_f1", "macro_f1", "acc"], 
                help='Metric to optimize')
    parser.add_argument('-t', '--optimizer_type', default='hyperparams', choices=['hyperparams', 'preprocessing'])
    
    args = parser.parse_args()
    
    opt = Optimizer(args.dataset, args.metric)
    best_hparams = opt.best_hparams
    if args.optimizer_type == 'hyperparams':
        hof_model = opt.optimize_model()
        print("hof_model:",hof_model)
        
        new_value = abs(opt.study_model.best_value)
        json_value = float(best_hparams["{}".format(opt.metric)]) if best_hparams["{}".format(opt.metric)] else 0
        
        if new_value > json_value:
            synt_embeddings = opt.synt_embeddings[hof_model["synt_embeddings"]]
            max_epoch = hof_model["max_epoch"]
            batch_size = hof_model["batch_size"]
            lr, max_length = hof_model["lr"], hof_model["max_length"]
            
            best_hparams["synt_embeddings"] = synt_embeddings
            best_hparams["max_epoch"] = max_epoch
            best_hparams["batch_size"] = batch_size
            best_hparams["lr"] = lr
            best_hparams["max_length"] = max_length
            best_hparams["{}".format(opt.metric)] = abs(opt.study_model.best_value)
            json_object = json.dumps(best_hparams, indent=4)
            
            with open(BEST_HPARAMS_FILE_PATH.format(args.dataset), 'w') as out_f:
                out_f.write(json_object)
    elif args.optimizer_type == 'preprocessing':
        hof_preprocessing = opt.optimize_preprocessing()
        print("hof_preprocessing:",hof_preprocessing)
            
        preprocessing = opt.preprocessing[hof_preprocessing["preprocessing"]]
        
        new_value = abs(opt.study_model.best_value)
        json_value = float(best_hparams["{}".format(opt.metric)]) if best_hparams["{}".format(opt.metric)] else 0
        
        if new_value > json_value:
            best_hparams["preprocessing"] = preprocessing
            best_hparams["{}".format(opt.metric)] = new_value
            json_object = json.dumps(best_hparams, indent=4)
            
            with open(BEST_HPARAMS_FILE_PATH.format(args.dataset), 'w') as out_f:
                out_f.write(json_object)
        
    model = 'bert'
    preprocessing = best_hparams["preprocessing"]
    synt_embeddings = best_hparams["synt_embeddings"]
    embedding = 'deepset/sentence_bert' if opt.dataset == 'semeval2010' else 'allenai/scibert_scivocab_uncased'
    max_epoch = best_hparams["max_epoch"]
    batch_size = best_hparams["batch_size"]
    lr, max_length = best_hparams["lr"], best_hparams["max_length"]
    
    print("Optimized parameters for dataset {}:".format(args.dataset))
    print("Preprocessing - {}; Model - {}; Embedding - {}; Synt Embedding.".format(preprocessing, model, embedding, synt_embeddings))
    print("Batch size - {};".format(batch_size))
    print("Learning rate - {}; Max Length - {}; Max epoch - {}.".format(lr, max_length, max_epoch))
    print("Best {}:".format(opt.data["optimize"]), abs(opt.study_model.best_value))
    