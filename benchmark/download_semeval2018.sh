mkdir benchmark/raw_semeval20181-1
mkdir benchmark/raw_semeval20181-2

wget -P benchmark/raw_semeval20181-1/Train https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.1.text.xml
wget -P benchmark/raw_semeval20181-1/Train https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.1.relations.txt
wget -P benchmark/raw_semeval20181-1/Train https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.2.text.xml
wget -P benchmark/raw_semeval20181-1/Train https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.2.relations.txt

wget -P benchmark/raw_semeval20181-2/Train https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.1.text.xml
wget -P benchmark/raw_semeval20181-2/Train https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.1.relations.txt
wget -P benchmark/raw_semeval20181-2/Train https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.2.text.xml
wget -P benchmark/raw_semeval20181-2/Train https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.2.relations.txt

wget -P benchmark/raw_semeval20181-1/Test https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.1.test.text.xml
wget -P benchmark/raw_semeval20181-1/Test https://lipn.univ-paris13.fr/~gabor/semeval2018task7/keys.test.1.1.txt

wget -P benchmark/raw_semeval20181-2/Test https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.2.test.text.xml
wget -P benchmark/raw_semeval20181-2/Test https://lipn.univ-paris13.fr/~gabor/semeval2018task7/keys.test.1.2.txt

python opennre/dataset/converters/converter_semeval2018.py --nlp_tool_type scientific
rm -r benchmark/raw_semeval20181-1
rm -r benchmark/raw_semeval20181-2