if [ -z "$NLP_TOOL" ]
  then
    NLP_TOOL="stanza"
fi

if [ -z "$NLP_MODEL" ]
  then
    NLP_MODEL="general"
fi

mkdir benchmark/raw_semeval2010
wget -P benchmark/raw_semeval2010 https://raw.githubusercontent.com/sahitya0000/Relation-Classification/master/corpus/SemEval2010_task8_training/TRAIN_FILE.TXT
wget -P benchmark/raw_semeval2010 https://raw.githubusercontent.com/sahitya0000/Relation-Classification/master/corpus/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT
python opennre/dataset/converters/converter_semeval2010.py --nlp_tool $NLP_TOOL --nlp_model $NLP_MODEL
rm -r benchmark/raw_semeval2010