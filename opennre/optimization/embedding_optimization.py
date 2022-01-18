import argparse
import json
import os

from opennre import config

from opennre.framework.train import Training

SYNT_EMBEDDINGS = [[0,0,0],[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,1,0],[1,0,1],[1,1,1]]

class EmbeddingOptimization():
    def __init__(self, dataset, metric):
        self.dataset = dataset
        self.metric = metric
        
        if not os.path.exists(config.BEST_HPARAMS_FILE_PATH.format(dataset)):
            dict = config.HPARAMS
            dict["{}".format(self.metric)] = 0
            json_object = json.dumps(dict, indent=4)
            with open(config.BEST_HPARAMS_FILE_PATH.format(dataset), 'w') as f:
                f.write(json_object)
        self.best_hparams = {}
        with open(config.BEST_HPARAMS_FILE_PATH.format(dataset), 'r') as f:
            self.best_hparams = json.load(f)
            
    def embedding_training(self):
        parameters = self.best_hparams
        parameters["dataset"] = self.dataset
        parameters["metric"] = self.metric
        
        embedding_value = 0
        
        for embed in SYNT_EMBEDDINGS:
        
            parameters["synt_embedding"] = embed
            
            train = Training(self.dataset, self.metric, parameters,None)
            
            result = train.train()
            new_value = result[self.metric]
            
            if new_value > self.value:
                self.value = new_value
                self.best_result = result
                embed_type, self.value = embed, new_value
            
        return embed_type, self.value
    
    

    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', default="semeval2010", choices=config.DATASETS, 
                help='Dataset')
    parser.add_argument('-m','--metric', default="micro_f1", choices=config.METRICS, 
                help='Metric to optimize')
    args = parser.parse_args()
    
    with open(config.BEST_HPARAMS_FILE_PATH.format(args.dataset), 'r') as f:
        best_hparams = json.load(f)
    embed = EmbeddingOptimization(args.dataset, args.metric)
    best_result = embed.best_result
    embedding, new_value = embed.embedding_training()
    print("Type:", embedding, "Value:", new_value)
    
    best_hparams = {}
    with open(config.BEST_HPARAMS_FILE_PATH.format(args.dataset), 'r') as f:
        best_hparams = json.load(f)
        
    json_value = float(best_hparams["{}".format(args.metric)]) if best_hparams["{}".format(args.metric)] else 0
    
    
    if new_value > json_value:
        best_hparams["synt_embeddings"] = embedding
        best_hparams["acc"] = best_result["acc"]
        best_hparams["macro_f1"] = best_result["macro_f1"]
        best_hparams["micro_f1"] = best_result["micro_f1"]
        
        json_object = json.dumps(best_hparams, indent=4)
        
        with open(config.BEST_HPARAMS_FILE_PATH.format(args.dataset), 'w') as out_f:
            out_f.write(json_object)