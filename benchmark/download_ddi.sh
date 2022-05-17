if [ -z "$NLP_TOOL" ]
  then
    NLP_TOOL="spacy"
fi

if [ -z "$NLP_MODEL" ]
  then
    NLP_MODEL="scientific"
fi

mkdir benchmark/raw_ddi
wget -P benchmark/raw_ddi https://github.com/isegura/DDICorpus/blob/master/DDICorpus-2013.zip?raw=true -O benchmark/raw_ddi/DDICorpus-2013.zip
unzip benchmark/raw_ddi/DDICorpus-2013.zip -d benchmark/raw_ddi
python opennre/dataset/converters/converter_ddi.py --nlp_tool $NLP_TOOL --nlp_model $NLP_MODEL
rm -r benchmark/raw_ddi