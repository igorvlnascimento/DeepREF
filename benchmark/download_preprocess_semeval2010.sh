mkdir benchmark/raw_semeval
wget -P benchmark/raw_semeval https://raw.githubusercontent.com/sahitya0000/Relation-Classification/master/corpus/SemEval2010_task8_training/TRAIN_FILE.TXT
wget -P benchmark/raw_semeval https://raw.githubusercontent.com/sahitya0000/Relation-Classification/master/corpus/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT
python pre_processing/preprocess_semeval.py
rm -r benchmark/raw_semeval