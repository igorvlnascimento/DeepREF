if [ -z "$NLP_TOOL" ]
then
      NLP_TOOL="stanza"
fi

if [ -z "$NLP_MODEL" ]
then
      NLP_MODEL="scientific"
fi

wget -P benchmark --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dxzHQUrU6Za9d9Tib_jtj2s-B86EkqeS' -O benchmark/raw_semeval20181-1.zip
unzip benchmark/raw_semeval20181-1.zip -d benchmark
rm benchmark/raw_semeval20181-1.zip
python opennre/dataset/converters/converter_semeval20181-1.py --nlp_tool $NLP_TOOL --nlp_model $NLP_MODEL
rm -r benchmark/raw_semeval20181-1