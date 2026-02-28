qmkdir -p benchmark/raw_semeval20181-2/Train benchmark/raw_semeval20181-2/Test

wget -P benchmark/raw_semeval20181-2/Train https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.1.text.xml
wget -P benchmark/raw_semeval20181-2/Train https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.1.relations.txt
wget -P benchmark/raw_semeval20181-2/Train https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.2.text.xml
wget -P benchmark/raw_semeval20181-2/Train https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.2.relations.txt

wget -P benchmark/raw_semeval20181-2/Test https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.2.test.text.xml
wget -P benchmark/raw_semeval20181-2/Test https://lipn.univ-paris13.fr/~gabor/semeval2018task7/keys.test.1.2.txt

uv run python deepref/dataset/preprocessor/semeval2018_preprocessor.py --dataset semeval20181-2 --path benchmark/raw_semeval20181-2/
rm -r benchmark/raw_semeval20181-2