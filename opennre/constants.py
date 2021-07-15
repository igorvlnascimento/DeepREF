preprocessing_choices = ['none', 'punct_digit', 'punct_stop_digit', 'entity_blinding'] # TODO : Add NER
datasets_choices = ['semeval2010', 'semeval2018', 'ddi'] # TODO : Add TACRED
model_choices = ["cnn", "pcnn", "bert"]
pretrain_choices = ["bert-base-uncased", "dmis-lab/biobert-v1.1", "allenai/scibert_scivocab_uncased"]

CLASSES_SEM_EVAL = ['Component-Whole(e2,e1)',
            'Other',
            'Instrument-Agency(e2,e1)',
            'Member-Collection(e1,e2)',
            'Cause-Effect(e2,e1)',
            'Entity-Destination(e1,e2)',
            'Content-Container(e1,e2)',
            'Message-Topic(e1,e2)',
            'Product-Producer(e2,e1)',
            'Member-Collection(e2,e1)',
            'Entity-Origin(e1,e2)',
            'Cause-Effect(e1,e2)',
            'Component-Whole(e1,e2)',
            'Message-Topic(e2,e1)',
            'Product-Producer(e1,e2)',
            'Entity-Origin(e2,e1)',
            'Content-Container(e2,e1)',
            'Instrument-Agency(e1,e2)',
            'Entity-Destination(e2,e1)']

CLASSES_SEM_EVAL_2018 = ['usage', 'result', 'model-feature', 'part_whole', 'topic', 'compare']

CLASSES_DDI = ['advise',
            'effect',
            'mechanism',
            'int',
            'none']