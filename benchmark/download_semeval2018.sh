mkdir -p benchmark/raw_semeval20181-1/Train benchmark/raw_semeval20181-1/Test
mkdir -p benchmark/raw_semeval20181-2/Train benchmark/raw_semeval20181-2/Test

wget -P benchmark/raw_semeval20181-1/Train https://raw.githubusercontent.com/gkata/SemEval2018Task7/testing/1.1.text.xml & wget -P benchmark/raw_semeval20181-1/Train https://raw.githubusercontent.com/gkata/SemEval2018Task7/testing/1.1.relations.txt &
wget -P benchmark/raw_semeval20181-1/Test https://raw.githubusercontent.com/gkata/SemEval2018Task7/testing/1.1.test.text.xml & wget -P benchmark/raw_semeval20181-1/Test https://raw.githubusercontent.com/gkata/SemEval2018Task7/testing/1.1.test.relations.txt &

wget -P benchmark/raw_semeval20181-2/Train https://raw.githubusercontent.com/gkata/SemEval2018Task7/testing/1.2.text.xml & wget -P benchmark/raw_semeval20181-2/Train https://raw.githubusercontent.com/gkata/SemEval2018Task7/testing/1.2.relations.txt &
wget -P benchmark/raw_semeval20181-2/Test https://raw.githubusercontent.com/gkata/SemEval2018Task7/testing/1.2.test.text.xml & wget -P benchmark/raw_semeval20181-2/Test https://raw.githubusercontent.com/gkata/SemEval2018Task7/testing/1.2.test.relations.txt &

wait

uv run python deepref/dataset/preprocessor/semeval2018_preprocessor.py --path benchmark/raw_semeval20181-1/ 2>&1 | sed -u 's/^/[job1] /' & uv run python deepref/dataset/preprocessor/semeval2018_preprocessor.py --path benchmark/raw_semeval20181-2/ 2>&1 | sed -u 's/^/[job2] /' &

wait

rm -r benchmark/raw_semeval20181-1 & rm -r benchmark/raw_semeval20181-2 &