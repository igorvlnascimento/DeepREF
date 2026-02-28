import argparse
import re

import pandas as pd
from tqdm import tqdm

from deepref.dataset.preprocessor.dataset_preprocessor import DatasetPreprocessor

class SemEval2010Preprocessor(DatasetPreprocessor):
    def tag_sentence(self, line):
        _, sent = line.split('\t')

        sent = sent.strip()
        if sent[0] == '"':
            sent = sent[1:]
        if sent[-1] == '"':
            sent = sent[:-1]
        normal_sent = sent
        normal_sent = normal_sent.replace('<e1>', ' ')
        normal_sent = normal_sent.replace('</e1>', ' ')
        normal_sent = normal_sent.replace('<e2>', ' ')
        normal_sent = normal_sent.replace('</e2>', ' ')
        tagged_sentence = sent.replace('<e1>', ' ENTITYSTART ')
        tagged_sentence = tagged_sentence.replace('</e1>', ' ENTITYEND ')
        tagged_sentence = tagged_sentence.replace('<e2>', ' ENTITYOTHERSTART ')
        tagged_sentence = tagged_sentence.replace('</e2>', ' ENTITYOTHEREND ')
        tagged_sentence = self.remove_whitespace(tagged_sentence) # to get rid of additional white space
        return tagged_sentence

    def get_sentences(self, filepath):
        lines = []
        with open(filepath, 'r') as file:
            lines = list(file.readlines())

        for i in tqdm(range(0, len(lines), 4)):
            line = lines[i]
            relation = lines[i+1].strip()
            tagged_sentence = self.tag_sentence(line)

            yield tagged_sentence, relation

    def get_csv_rows(self, filepath):
        with open(filepath, 'r') as f:
            lines = list(f.readlines())
        for i in tqdm(range(0, len(lines), 4)):
            _, sent = lines[i].split('\t')
            sent = sent.strip()
            if sent[0] == '"':
                sent = sent[1:]
            if sent[-1] == '"':
                sent = sent[:-1]
            relation = lines[i + 1].strip()

            e1_match = re.search(r'<e1>(.*?)</e1>', sent)
            e2_match = re.search(r'<e2>(.*?)</e2>', sent)
            e1_name = e1_match.group(1).strip() if e1_match else ''
            e2_name = e2_match.group(1).strip() if e2_match else ''

            # Count words before each tag after removing the other entity's tags
            plain_no_e2 = re.sub(r'</?e2>', '', sent)
            e1_start = len(plain_no_e2[:plain_no_e2.find('<e1>')].split())
            e1_end = e1_start + len(e1_name.split())

            plain_no_e1 = re.sub(r'</?e1>', '', sent)
            e2_start = len(plain_no_e1[:plain_no_e1.find('<e2>')].split())
            e2_end = e2_start + len(e2_name.split())

            plain = re.sub(r'</?e[12]>', '', sent)
            plain = ' '.join(plain.split())

            yield {
                'original_sentence': plain,
                'e1': str({'name': e1_name, 'position': [e1_start, e1_end]}),
                'e2': str({'name': e2_name, 'position': [e2_start, e2_end]}),
                'relation_type': relation,
                'pos_tags': '', 'dependencies_labels': '', 'ner': '', 'sk_entities': ''
            }


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(description="Generate CSV from raw SemEval2010 files")
    parser.add_argument("--path", required=True, help="Path to directory containing raw SemEval2010 txt files")
    args = parser.parse_args()

    preprocessor = SemEval2010Preprocessor()
    data = (
        list(preprocessor.get_csv_rows(os.path.join(args.path, "TRAIN_FILE.TXT"))) +
        list(preprocessor.get_csv_rows(os.path.join(args.path, "TEST_FILE_FULL.TXT")))
    )
    df = pd.DataFrame(data)
    output = "benchmark/semeval2010.csv"
    df.to_csv(output, sep='\t', index=False)
    print(f"Written {len(df)} rows to {output}")
