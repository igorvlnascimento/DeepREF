if [ -z "$1" ]
then
      $1 = "stanza"
fi

if [ -z "$2" ]
then
      $2 = "general"
fi

mkdir benchmark/raw_semeval2010
wget -P benchmark/raw_semeval2010 https://raw.githubusercontent.com/sahitya0000/Relation-Classification/master/corpus/SemEval2010_task8_training/TRAIN_FILE.TXT
wget -P benchmark/raw_semeval2010 https://raw.githubusercontent.com/sahitya0000/Relation-Classification/master/corpus/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT
python opennre/dataset/converters/converter_semeval2010.py --nlp_tool $1 --nlp_tool_type $2
rm -r benchmark/raw_semeval2010