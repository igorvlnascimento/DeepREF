# OpenNRE++

An optimization framework for relation classification tasks

## What is Relation Extraction

Relation extraction is a natural language processing (NLP) task aiming at extracting relations (e.g., *founder of*) between entities (e.g., **Bill Gates** and **Microsoft**). For example, from the sentence *Bill Gates founded Microsoft*, we can extract the relation triple (**Bill Gates**, *founder of*, **Microsoft**). 

Relation extraction is a crucial technique in automatic knowledge graph construction. By using relation extraction, we can accumulatively extract new relation facts and expand the knowledge graph, which, as a way for machines to understand the human world, has many downstream applications like question answering, recommender system and search engine. 

### Using Git Repository

Clone the repository from our github page (don't forget to star us!)

```bash
git clone https://github.com/igorvlnascimento/open-nre-plus-plus.git
```

Then install all the requirements:

```
pip install -r requirements.txt
```

Then install the package with 
```
python setup.py install
```

## Preprocessing

It's easy to preprocess the datasets on OpenNRE++. Just execute the code below:

```
python opennre/dataset/preprocess_dataset.py -d <dataset>
```

Dataset can be one of theses options: 'semeval2010' (default), 'semeval20181-1', 'semeval20181-2' and 'ddi'.

This code will download the dataset, transform into a .csv file in a standard format and preprocess it to the choosen processing types. The preprocessing types can be a combination of 'eb' (entity blinding), 'nb' (NER blinding), 'd' (digit blinding), 'b' (text between brackets or parenthesis removal), 'p' (punctuation removal) and 'sw' (stopwords removal). If you leave the '-p' in blank, it will make all possible combination  with theses preprocessing types. 

For example, if yu do this:
```
python opennre/dataset/preprocess_dataset.py -d semeval2010 -p sw d
```

It will download SemEval 2010 dataset, if it didn't yet, transform to a .csv file in a standard format and preprocess to remove the stopwords and blind the digits on the dataset sentences. 

If you only wants to download the dataset and transform it to a .csv file in a standard format, only execute the following:
```
bash benchmark/download_<dataset>.sh
```

You can change the NLP tool and NLP tool type from 'opennre/data/best_hparams_<dataset>.json' file. The 'nlp_tool' can be only 'stanza' or 'spacy' and 'nlp_tool_type' is 'general' or 'scientific'. The 'general' type loads a more general model for Stanza or SpaCy and the 'scientific' loads the biomedical model for Stanza and the smae general model for SpaCy.

This can take from about 30 minutes to 3 hours to execute for Stanza (depending on the dataset) and less than 10 minutes for SpaCy. But Stanza has more accuracy.

## Training

Make sure you have installed OpenNRE++ as instructed above. Then import our package and load pre-trained models.

If you want to train one or a few models, try this:
```
python opennre/framework/train.py --dataset <dataset> --metric <metric>
```

Even if you don't have any datasets preprocessed, the code above can automatically download and preprocess the dataset for you on the fly.

Possible metrics are: 'micro_f1' (default), 'macro_f1' and 'acc'.

That code will get the hyperparameters, embeddings and preprocessing types from 'opennre/data/best_hparams_<dataset>.json'. If this file doesn't exist, it will create it automatically with default values. This file contains the best hyperparameters, embeddings and preprocessing type for such dataset. You can change the values manually to get a different result. The list of possible values you can use to test can be seen on `opennre/constants.py`. You can see the results in the file `results/<dataset>/ResultsOpenNRE++_<dataset>_<datetime>.txt`. It can take about 20 minutes to execute the training on Colab Pro+ using `GPU` and `High RAM`.

## Optimization

### Hyperparameters/Model optimization

To optimize hyperparameters or models, execute the following code:
```
python opennre/optimization/optuna_optimizer.py -d <dataset> -m <metric> -t <trials_number> -o <optimization_type>
```

The first two are the datasets and metrics already explained above. The '-t' arg means the number of trials that Optuna needs to execute to find the best combination of parameters. The '-o' is the type of optimization that you can choose between `hyperparams` or `model`. If you choose `hyperparams` (default), Optuna will find the best hyperparameters for the model and embedding indicated on 'best_hparams_<dataset>.json' file. The default model and embedding are `bert` and `bert-base-uncaseed`, respectively. If you choose the `model` option, Optuna will find the best model and embedding combination. The possible models and embeddings to be choosen can be seen on `opennre/constants.py` file. 

### Preprocessing type optimization

If you wish to get the best preprocessing type combination for the indicated model and embedding on 'best_hparams_<dataset>.json' file, execute the following code:
```
python opennre/optimization/preprocess_optimization.py -d <dataset> -m <metric>
```

It will call Optuna to find the best preprocessing type for the choosen model. Optuna has the advantage to use a pruning algorithm to stop execution when the training intermediate results are below the past execution results average.

### Semantic and syntatic embeddings optimization

We implement 3 types of embeddings: semantic knowledge (SK) embedding, part-of-speech tags (POS) embedding and dependency graphs (deps) embedding. We implement an optimization code to find the best combination of theses embeddings for each dataset. 