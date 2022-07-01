if [ -z "$NLP_TOOL" ]
  then
    NLP_TOOL="spacy"
fi

if [ -z "$NLP_MODEL" ]
  then
    NLP_MODEL="en_core_web_sm"
fi

mkdir benchmark/raw_semeval2010
wget -P benchmark/raw_semeval2010 https://raw.githubusercontent.com/sahitya0000/Relation-Classification/master/corpus/SemEval2010_task8_training/TRAIN_FILE.TXT
wget -P benchmark/raw_semeval2010 https://raw.githubusercontent.com/sahitya0000/Relation-Classification/master/corpus/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT
python opennre/dataset/converters/semeval2010_converter.py --nlp_tool $NLP_TOOL --nlp_model $NLP_MODEL --train_path benchmark/raw_semeval2010/TRAIN_FILE.TXT --test_path benchmark/raw_semeval2010/TEST_FILE_FULL.TXT
rm -r benchmark/raw_semeval2010