mkdir -p benchmark/raw_semeval2010
wget -P benchmark/raw_semeval2010 https://raw.githubusercontent.com/sahitya0000/Relation-Classification/master/corpus/SemEval2010_task8_training/TRAIN_FILE.TXT & wget -P benchmark/raw_semeval2010 https://raw.githubusercontent.com/sahitya0000/Relation-Classification/master/corpus/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT &

wait

uv run python deepref/dataset/preprocessor/semeval2010_preprocessor.py --path benchmark/raw_semeval2010/
rm -r benchmark/raw_semeval2010