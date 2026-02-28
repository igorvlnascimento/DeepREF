import argparse
import re

import pandas as pd
from tqdm import tqdm

from deepref.dataset.preprocessor.dataset_preprocessor import DatasetPreprocessor
from deepref.nlp.spacy_nlp_tool import SpacyNLPTool

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

    def get_entity_dict(self, *args):
        pass


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(description="Generate CSV from raw SemEval2010 files")
    parser.add_argument("--path", required=True, help="Path to directory containing raw SemEval2010 txt files")
    args = parser.parse_args()

    preprocessor = SemEval2010Preprocessor()
    data = (
        list(preprocessor.get_sentences(os.path.join(args.path, "TRAIN_FILE.TXT"))) +
        list(preprocessor.get_sentences(os.path.join(args.path, "TEST_FILE_FULL.TXT")))
    )
    tool = SpacyNLPTool("en_core_web_trf")

    preprocessor.write_csv("semeval2010", data, tool)
