import argparse
import json
import os

from opennre import constants

from opennre.framework.train import Training

SYNT_EMBEDDINGS = [[0,0,0],[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,1,0],[1,0,1],[1,1,1]]

class EmbeddingOptimization():
    def __init__(self, dataset, metric):
        self.dataset = dataset
        self.metric = metric
        
        if not os.path.exists(constants.BEST_HPARAMS_FILE_PATH.format(dataset)):
            dict = {
                "{}".format(self.metric): 0,
                "model": "bert",
                "embedding": "bert-base-uncased",
                "batch_size": 16,
                "preprocessing": 0,
                "lr": 2e-5,
                "synt_embeddings": [0,0,0],
                "max_length": 128,
                "max_epoch": 3
            }
            json_object = json.dumps(dict, indent=4)
            with open(constants.BEST_HPARAMS_FILE_PATH.format(dataset), 'w') as f:
                f.write(json_object)
        self.best_hparams = {}
        with open(constants.BEST_HPARAMS_FILE_PATH.format(dataset), 'r') as f:
            self.best_hparams = json.load(f)
            
    def embedding_training(self):
        parameters = self.best_hparams
        parameters["dataset"] = self.dataset
        parameters["metric"] = self.metric
        
        for embed in SYNT_EMBEDDINGS:
        
            parameters["synt_embedding"] = embed
            
            train = Training(parameters,None)
            
            new_value = train.train()
            
            if new_value > embedding_value:
                embed_type, embedding_value = embed, new_value 
            
        return embed_type, embedding_value
    
    

    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', default="semeval2010", choices=["semeval2010", "semeval2018", "semeval20181-1", "semeval20181-2", "ddi"], 
                help='Dataset')
    parser.add_argument('-m','--metric', default="micro_f1", choices=["micro_f1", "macro_f1", "acc"], 
                help='Metric to optimize')
    args = parser.parse_args()
    dataset = args.dataset
    metric = args.metric
    embed = EmbeddingOptimization(dataset, metric)
    embedding, new_value = embed.embedding_training()
    print("Type:", embedding, "Value:", new_value)
    
    best_hparams = {}
    with open(constants.BEST_HPARAMS_FILE_PATH.format(dataset), 'r') as f:
        best_hparams = json.load(f)
        
    json_value = float(best_hparams["{}".format(metric)]) if best_hparams["{}".format(metric)] else 0
    
    if new_value > json_value:
        best_hparams["synt_embeddings"] = embedding
        best_hparams["{}".format(metric)] = new_value
        best_hparams.pop("dataset",None)
        best_hparams.pop("metric",None)
        
        json_object = json.dumps(best_hparams, indent=4)
        
        with open(constants.BEST_HPARAMS_FILE_PATH.format(dataset), 'w') as out_f:
            out_f.write(json_object)
    #print(preprocessing[30])