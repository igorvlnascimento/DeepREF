mkdir benchmark/raw_ddi
wget -P benchmark/raw_ddi https://github.com/isegura/DDICorpus/blob/master/DDICorpus-2013.zip?raw=true -O benchmark/raw_ddi/DDICorpus-2013.zip
unzip benchmark/raw_ddi/DDICorpus-2013.zip -d benchmark/raw_ddi
python pre_processing/preprocess_ddi.py
rm -r benchmark/raw_ddi