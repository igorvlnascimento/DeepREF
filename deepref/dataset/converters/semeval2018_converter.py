import argparse
from tqdm import tqdm
from pyexpat import ExpatError
import xml.etree.ElementTree as ET
from pathlib import Path

from deepref import config
from deepref.dataset.converters.dataset_converter import DatasetConverter

from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

class SemEval2018Converter(DatasetConverter):
    def __init__(self, dataset_name, nlp_tool, nlp_model):
        super().__init__(dataset_name, nlp_tool=nlp_tool, nlp_model=nlp_model)
        
    def get_entity_dict(self, text_elements):
        punk_param = PunktParameters()
        abbreviations = ['e.g', 'viz', 'al']
        punk_param.abbrev_types = set(abbreviations)
        tokenizer = PunktSentenceTokenizer(punk_param)
        for text in text_elements:
            entity_dict = {}
            sentences = ''
            char_offset_start = 0
            char_offset_end = 0
            if text.text is not None:
                sentences += text.text.lstrip()
                char_offset_start += len(text.text.lstrip())    
            for entity in text:
                if entity.tag == 'entity' and entity.text is not None:
                    sentences += entity.text
                    char_offset_start = sentences.rfind(entity.text)
                    char_offset_end = sentences.rfind(entity.text) + len(entity.text) - 1
                    char_offset = [f'{char_offset_start}-{char_offset_end}']
                    entity_dict[entity.get('id')] = {'word': entity.text, 'charOffset': char_offset}
                    assert entity.text == sentences[char_offset_start:char_offset_end+1]
                    if entity.tail is not None:
                        sentences += entity.tail
                    
                    
            sentences = tokenizer.tokenize(sentences)
            for i, s in enumerate(sentences):
                sentences_length = len(" ".join(sentences[:i])) if len(" ".join(sentences[:i])) == 0 else len(" ".join(sentences[:i])) + 1
                entity_dict_sentence = {}
                for e in entity_dict:
                    positions = [int(entity_dict[e]['charOffset'][0].split('-')[0]), int(entity_dict[e]['charOffset'][0].split('-')[1])]
                    if positions[0] >= sentences_length and positions[1] <= len(s) + sentences_length:
                        entity_dict_sentence[e] = {'word': entity_dict[e]['word']}
                        char_offset_start = int(entity_dict[e]['charOffset'][0].split('-')[0]) - sentences_length
                        char_offset_end = int(entity_dict[e]['charOffset'][0].split('-')[1]) - sentences_length
                        entity_dict_sentence[e]['charOffset'] = [f'{char_offset_start}-{char_offset_end}']
                yield s, entity_dict_sentence
    
    def get_entity_pairs(self, path):
        entity_pairs = {}
        for filepath in Path(path).rglob('*.txt'):
            lines = open(filepath).readlines()
            for line in lines:
                relation = line[:line.find('(')]
                if 'REVERSE' in line:
                    e2_id = line[line.find('(')+1:line.find(',')]
                    e1_id = line[line.find(',')+1:line.find(',REVERSE)')]
                    entity_pairs[e1_id] = {'relation': relation, 'e1': e1_id, 'e2':e2_id}
                else:
                    e1_id = line[line.find('(')+1:line.find(',')]
                    e2_id = line[line.find(',')+1:line.find(')')]
                    entity_pairs[e1_id] = {'relation': relation, 'e1': e1_id, 'e2':e2_id}
        return entity_pairs
    
    def get_sentences(self, path):
        for filepath in Path(path).rglob('*.xml'):
            try:
                tree = ET.parse(str(filepath))
                root = tree.getroot()
            except ExpatError:
                pass
            
            text_elements = root.findall('./text/')
            
            for sentence, entity_dict in tqdm(self.get_entity_dict(text_elements)):
                for e1_id in entity_dict:
                    pairs = self.get_entity_pairs(path)
                    if e1_id in pairs:
                        e2_id = pairs[e1_id]['e2']
                        relation = pairs[e1_id]['relation'].lower()
                        
                        other_entities = self.get_other_entities(entity_dict, e1_id, e2_id)
                        e1_data = entity_dict[e1_id]
                        if e2_id in entity_dict:
                            e2_data = entity_dict[e2_id]
                        else:
                            continue
                        tagged_sentence = self.tag_sentence(sentence, e1_data, e2_data, other_entities)
                            
                        yield tagged_sentence, relation
                            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='semeval20181-1',
        help='Dataset name')
    parser.add_argument('--train_path', default='benchmark/raw_semeval20181-1/Train/', 
        help='Input path of training examples')
    parser.add_argument('--test_path', default='benchmark/raw_semeval20181-1/Test/', 
        help='Input path of test examples')
    parser.add_argument('--nlp_tool', default='spacy', choices=config.NLP_TOOLS,
        help='NLP tool name')
    parser.add_argument('--nlp_model', default='en_core_web_sm',
        help='NLP tool model name')

    args = parser.parse_args()
    
    converter = SemEval2018Converter(args.dataset, args.nlp_tool, args.nlp_model)
    converter.create_dataset(converter.get_sentences(args.train_path), converter.get_sentences(args.test_path))
