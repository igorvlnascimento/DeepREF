import itertools
import json
import os

from train import Training

CONFIG_FILE_PATH = "opennre/optimization/config_params.json"
BEST_HPARAMS_FILE_PATH = "opennre/optimization/best_hparams_{}.json"

class EmbeddingOptimization():
    def __init__(self, dataset, metric):
        self.dataset = dataset
        self.metric = metric
        self.data = json.load(open(CONFIG_FILE_PATH))
        #self.preprocessing = self.data["preprocessing"]
        #self.preprocess_combination = self.combine_preprocessing(self.preprocessing)
        
        # synt_embeddings = [0,0,0]
        # if dataset == 'semeval2010':
        #     synt_embeddings = [1,1,1]
        # elif dataset == 'ddi':
        #     synt_embeddings = [1,0,1]
        # elif dataset == 'semeval20181-1':
        #     synt_embeddings = [1,1,0]
        # elif dataset == 'semeval20181-2':
        #     synt_embeddings = [1,0,1]
        self.embeddings = [[0,0,0],[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,1,0],[1,0,1],[1,1,1]]
        
        if not os.path.exists(BEST_HPARAMS_FILE_PATH.format(dataset)):
            dict = {
                "{}".format(self.metric): 0,
                "batch_size": 16,
                "preprocessing": 0,
                "lr": 2e-5,
                "synt_embeddings": [0,0,0],
                "max_length": 128,
                "max_epoch": 3
            }
            json_object = json.dumps(dict, indent=4)
            with open(BEST_HPARAMS_FILE_PATH.format(dataset), 'w') as f:
                f.write(json_object)
        self.best_hparams = {}
        with open(BEST_HPARAMS_FILE_PATH.format(dataset), 'r') as f:
            self.best_hparams = json.load(f)

    # def permutation_embedding(self):
            
    #         final_permutations = list(itertools.permutations(range(2), 3))
    #         print(final_permutations)
    #         return final_permutations
        
    def embedding_training(self):
        model = 'bert',#self.study_model.best_params["model"]
        pretrain_bert = 'bert-base-uncased' if self.dataset == 'semeval2010' else 'allenai/scibert_scivocab_uncased'#individual.suggest_categorical("pretrain_bert", self.data["pretrain_bert"])
        synt_embeddings = self.best_hparams["synt_embeddings"]

        batch_size =  2
        lr =  self.best_hparams["lr"]
        max_length = self.best_hparams["max_length"]
        max_epoch = self.best_hparams["max_epoch"]
        
        embed_type, embedding_value = synt_embeddings, 0
        
        for embed in self.embeddings:
        
            parameters = {
                "dataset": self.dataset,
                "model": model,
                "metric": self.metric,
                "preprocessing": [],
                "embedding": pretrain_bert,
                "synt_embeddings": embed,
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
            
            if new_value > embedding_value:
                embed_type, embedding_value = embed, new_value 
            
        return embed_type, embedding_value
    
    

    

    
if __name__ == '__main__':
    dataset = "semeval2010"
    metric = "micro_f1"
    embed = EmbeddingOptimization(dataset, metric)
    embedding, new_value = embed.embedding_training()
    print("Type:", embedding, "Value:", new_value)
    
    best_hparams = {}
    with open(BEST_HPARAMS_FILE_PATH.format(dataset), 'r') as f:
        best_hparams = json.load(f)
        
    json_value = float(best_hparams["{}".format(metric)]) if best_hparams["{}".format(metric)] else 0
    
    if new_value > json_value:
        best_hparams["synt_embeddings"] = embedding
        #best_hparams["synt_embeddings"] = synt_embeddings
        best_hparams["{}".format(metric)] = new_value
        json_object = json.dumps(best_hparams, indent=4)
        
        with open(BEST_HPARAMS_FILE_PATH.format(dataset), 'w') as out_f:
            out_f.write(json_object)
    #print(preprocessing[30])