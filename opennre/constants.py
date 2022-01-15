import itertools

def combine_preprocessing(preprocessing):
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
        return final_combinations

SEED = 42
METRICS = ["micro_f1", "macro_f1", "acc"]
PREPROCESSING_TYPES = ["sw", "d", "b", "p", "eb", "nb"]
PREPROCESSING_COMBINATION = combine_preprocessing(PREPROCESSING_TYPES)
DATASETS = ['semeval2010', 'semeval20181-1', 'semeval20181-2', 'ddi'] # TODO : Add TACRED
MODELS = ["cnn", "pcnn", "crcnn", "gru", "bigru", "lstm", "bilstm", "bert"]
PRETRAIN_WEIGHTS = ["bert-base-uncased", "dmis-lab/biobert-v1.1", "allenai/scibert_scivocab_uncased"]
NLP_TOOLS = ["stanza", "spacy"]
NLP_TOOLS_TYPE = ["general", "scientific"]

########################### PATHS #######################################
RESULTS_PATH = "results"
NLP_CONFIG = "opennre/data/nlp_config.json"
RELATIONS_TYPE = "opennre/data/relations_type.json"
CLASSES_DATASET = "opennre/data/classes_dataset.json"
BEST_HPARAMS_FILE_PATH = "opennre/data/best_hparams_{}.json"