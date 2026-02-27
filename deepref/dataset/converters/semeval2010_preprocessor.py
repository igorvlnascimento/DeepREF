import argparse
from tqdm import tqdm
from deepref import config

from deepref.dataset.converters.dataset_converter import DatasetConverter

class SemEval2010Converter(DatasetConverter):
    def __init__(self, nlp_tool, nlp_model):
        super().__init__(dataset_name='semeval2010', nlp_tool=nlp_tool, nlp_model=nlp_model)
        
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nlp_tool', default='spacy', choices=config.NLP_TOOLS,
        help='NLP tool name')
    parser.add_argument('--nlp_model', default='en_core_web_sm', 
        help='NLP tool model name')
    parser.add_argument('--train_path', default='benchmark/raw_semeval2010/TRAIN_FILE.TXT', 
        help='Train filepath')
    parser.add_argument('--test_path', default='benchmark/raw_semeval2010/TEST_FILE_FULL.TXT', 
        help='Test filepath')

    args = parser.parse_args()
    
    converter = SemEval2010Converter(args.nlp_tool, args.nlp_model)
    converter.create_dataset(converter.get_sentences(args.train_path), converter.get_sentences(args.test_path))