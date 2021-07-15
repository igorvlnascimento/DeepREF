mkdir benchmark/raw_semeval2018
mkdir benchmark/raw_semeval2018/Train
mkdir benchmark/raw_semeval2018/Test
wget -P benchmark/raw_semeval2018/Train https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.1.text.xml
wget -P benchmark/raw_semeval2018/Train https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.1.relations.txt
wget -P benchmark/raw_semeval2018/Test https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.1.test.text.xml
wget -P benchmark/raw_semeval2018/Test https://lipn.univ-paris13.fr/~gabor/semeval2018task7/keys.test.1.1.txt
#python pre_processing/preprocess_semeval2018.py
#rm -r benchmark/raw_semeval2018