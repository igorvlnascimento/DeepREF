if [ -z "$NLP_TOOL" ]
then
      NLP_TOOL="spacy"
fi

if [ -z "$NLP_MODEL" ]
then
      NLP_MODEL="en_core_web_sm"
fi

wget -P benchmark --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dxzHQUrU6Za9d9Tib_jtj2s-B86EkqeS' -O benchmark/raw_semeval20181-1.zip
unzip benchmark/raw_semeval20181-1.zip -d benchmark
rm benchmark/raw_semeval20181-1.zip
python opennre/dataset/converters/semeval2018_converter.py --dataset semeval20181-1 --train_path benchmark/raw_semeval20181-1/Train/ --test_path benchmark/raw_semeval20181-1/Test/ --nlp_tool $NLP_TOOL --nlp_model $NLP_MODEL
rm -r benchmark/raw_semeval20181-1