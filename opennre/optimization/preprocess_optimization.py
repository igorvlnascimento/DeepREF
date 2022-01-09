import itertools
import json

from train import Training

CONFIG_FILE_PATH = "opennre/optimization/config_params.json"

class PreprocessOptimization():
    def __init__(self, dataset, metric):
        self.dataset = dataset
        self.metric = metric
        self.data = json.load(open(CONFIG_FILE_PATH))
        self.preprocessing = self.data["preprocessing"]
        self.preprocess_combination = self.combine_preprocessing(self.preprocessing)

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
        synt_embeddings = [1,1,1]

        batch_size =  2
        lr =  2e-5
        max_length =  128
        max_epoch = 3
        
        preprocessing_type, preprocessing_value = 0, 0
        
        for i in range(len(self.preprocessing)):
        
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
                preprocessing_type, preprocessing_value = self.preprocess_combination[i], new_value 
            
        return preprocessing_type, preprocessing_value
    
    

    

    
if __name__ == '__main__':
    prep = PreprocessOptimization('semeval2010', 'micro-f1')
    type, value = prep.preprocessing_training()
    print("Type:", type, "Value:", value)
    #print(preprocessing[30])