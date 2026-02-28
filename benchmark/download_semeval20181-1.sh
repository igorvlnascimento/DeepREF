wget -P benchmark --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dxzHQUrU6Za9d9Tib_jtj2s-B86EkqeS' -O benchmark/raw_semeval20181-1.zip
unzip benchmark/raw_semeval20181-1.zip -d benchmark
rm benchmark/raw_semeval20181-1.zip
uv run python deepref/dataset/preprocessor/semeval2018_preprocessor.py --dataset semeval20181-1 --path benchmark/raw_semeval20181-1/
rm -r benchmark/raw_semeval20181-1