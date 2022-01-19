if [ -z "$1" ]
then
      $1 = "stanza"
fi

if [ -z "$2" ]
then
      $2 = "scientific"
fi

wget -P benchmark --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dxzHQUrU6Za9d9Tib_jtj2s-B86EkqeS' -O benchmark/raw_semeval20181-1.zip
unzip benchmark/raw_semeval20181-1.zip -d benchmark
rm benchmark/raw_semeval20181-1.zip
python opennre/dataset/converters/converter_semeval20181-1.py --nlp_tool $1 --nlp_model $2
rm -r benchmark/raw_semeval20181-1