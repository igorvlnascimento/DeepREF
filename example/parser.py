import subprocess

preprocessing_choices = ['none', 'punct_digit', 'punct_stop_digit', 'entity_blinding']

for preprocessing in preprocessing_choices:
    subprocess.call(['python', 'example/train_supervised_roberta.py', '--dataset', 'semeval2010', '--preprocessing', preprocessing])