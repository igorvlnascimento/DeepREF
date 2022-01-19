if [ -z "$1" ]
then
      $1 = "stanza"
fi

if [ -z "$2" ]
then
      $2 = "scientific"
fi

mkdir benchmark/raw_ddi
wget -P benchmark/raw_ddi https://github.com/isegura/DDICorpus/blob/master/DDICorpus-2013.zip?raw=true -O benchmark/raw_ddi/DDICorpus-2013.zip
unzip benchmark/raw_ddi/DDICorpus-2013.zip -d benchmark/raw_ddi
python opennre/dataset/converters/converter_ddi.py --nlp_tool $1 --nlp_model $2
rm -r benchmark/raw_ddi