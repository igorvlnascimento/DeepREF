import itertools

def combine(list_, type_=''):
        combinations = []
        for i in range(len(list_)+1):
            combinations.extend(itertools.combinations(list_, i))
            
        if type_ == 'preprocessing':
            for j, comb in enumerate(combinations):
                if 'eb' in comb and 'nb' in comb:
                    comb = list(comb)
                    comb.remove('eb')
                    combinations[j] = comb
                else:
                    combinations[j] = list(comb)
        else:
            combinations = [" ".join(list(comb)) for comb in combinations]
        
        final_combinations = [comb for n, comb in enumerate(combinations) if comb not in combinations[:n]]
        return final_combinations

SEED = 42
METRICS = ["micro_f1", "macro_f1", "acc"]
PREPROCESSING_TYPES = ["sw", "d", "b", "p", "eb", "nb"]
PREPROCESSING_COMBINATION = combine(PREPROCESSING_TYPES, 'preprocessing')
DATASETS = ['semeval2010', 'semeval20181-1', 'semeval20181-2', 'ddi'] # TODO : Add TACRED
MODELS = ["cnn", "pcnn", "crcnn", "gru", "bigru", "lstm", "bilstm", "bert_cls", "bert_entity"]
PRETRAIN_WEIGHTS = ["bert-base-uncased", "dmis-lab/biobert-v1.1", "allenai/scibert_scivocab_uncased", "deepset/sentence_bert"]
WORD_EMBEDDINGS = ["glove", "senna", "fasttext_wiki", "fasttext_crawl"]
TYPE_EMBEDDINGS = ["position", "sk", "pos_tags", "deps"]
TYPE_EMBEDDINGS_COMBINATION = combine(TYPE_EMBEDDINGS)
NLP_TOOLS = ["stanza", "spacy"]
NLP_MODEL = ["general", "scientific"]
HPARAMS = {
    "model": "bert_entity",
    "pretrain": "bert-base-uncased",
    "batch_size": 16,
    "preprocessing": [],
    "lr": 2e-5,
    "position_embed": 0,
    "pos_tags_embed": 0,
    "deps_embed": 0,
    "sk_embed": 0,
    "sdp_embed": 0,
    "max_length": 128,
    "max_epoch": 1
}

########################### PATHS #######################################
RESULTS_PATH = "results"
NLP_CONFIG = "opennre/data/nlp_config.json"
RELATIONS_TYPE = "opennre/data/relations_type.json"
DATASET_CLASSES = "opennre/data/dataset_classes.json"
BEST_HPARAMS_FILE_PATH = "opennre/data/best_hparams_{}.json"