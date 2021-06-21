# OpenNRE++

A more complete 

## What is Relation Extraction

Relation extraction is a natural language processing (NLP) task aiming at extracting relations (e.g., *founder of*) between entities (e.g., **Bill Gates** and **Microsoft**). For example, from the sentence *Bill Gates founded Microsoft*, we can extract the relation triple (**Bill Gates**, *founder of*, **Microsoft**). 

Relation extraction is a crucial technique in automatic knowledge graph construction. By using relation extraction, we can accumulatively extract new relation facts and expand the knowledge graph, which, as a way for machines to understand the human world, has many downstream applications like question answering, recommender system and search engine. 

## Install 

### Install as A Python Package

We are now working on deploy OpenNRE as a Python package. Coming soon!

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

## Training

Make sure you have installed OpenNRE as instructed above. Then import our package and load pre-trained models.

This code below train all the inputs. It can take several hours and you need a very good GPU:
```
python example/parser.py
```

If you want to train one or a few models, try this:
```
python example/train_supervised_<model> --dataset <dataset> --preprocessing <preprocessing> --pretrain_path <pretrain_path>
```

Which model = ['cnn', 'bert'], dataset = ['semeval2010', 'ddi'], preprocessing = ['none', 'punct_digit', 'punct_stop_digit', 'entity_blinding'], pretrain_path = ['bert-base-uncased', 'dmis-lab/biobert-v1.1', 'allenai/scibert_scivocab_uncased']

For example:
```
python example/train_supervised_bert --dataset semeval2010 --preprocessing punct_digit --pretrain_path bert-base-uncased
```

For 'cnn' you should use the default 'glove' for <pretrain_path>

You can train your own models on your own data with OpenNRE++. In `example` folder we give example training codes for supervised RE models and bag-level RE models. You can either use our provided datasets or your own datasets.
