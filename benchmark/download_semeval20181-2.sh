if [ -z "$NLP_TOOL" ]
then
      NLP_TOOL="stanza"
fi
echo $NLP_TOOL
if [ -z "$NLP_MODEL" ]
then
      NLP_MODEL="scientific"
fi

mkdir benchmark/raw_semeval20181-2

wget -P benchmark/raw_semeval20181-2/Train https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.1.text.xml
wget -P benchmark/raw_semeval20181-2/Train https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.1.relations.txt
wget -P benchmark/raw_semeval20181-2/Train https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.2.text.xml
wget -P benchmark/raw_semeval20181-2/Train https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.2.relations.txt

wget -P benchmark/raw_semeval20181-2/Test https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.2.test.text.xml
wget -P benchmark/raw_semeval20181-2/Test https://lipn.univ-paris13.fr/~gabor/semeval2018task7/keys.test.1.2.txt

python opennre/dataset/converters/converter_semeval20181-2.py --nlp_tool $NLP_TOOL --nlp_model $NLP_MODEL
rm -r benchmark/raw_semeval20181-2