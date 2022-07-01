import argparse
from pyexpat import ExpatError
from xml.dom import minidom
from pathlib import Path
from tqdm import tqdm

from deepref import config
from deepref.dataset.converters.dataset_converter import DatasetConverter

class DDIConverter(DatasetConverter):
    def __init__(self, nlp_tool, nlp_model):
        super().__init__(dataset_name="ddi", nlp_tool=nlp_tool, nlp_model=nlp_model)
        
    def get_entity_dict(self, sentence_dom):
        entities = sentence_dom.getElementsByTagName('entity')
        entity_dict = {}
        for entity in entities:
            id = entity.getAttribute('id')
            word = entity.getAttribute('text')
            charOffset = entity.getAttribute('charOffset')
            charOffset = charOffset.split(';') # charOffset can either be length 1 or 2 
            entity_dict[id] = {'word': word, 'charOffset': charOffset}
        return entity_dict
    
    def get_sentences(self, path):
        files_path = [filepath for filepath in Path(path).rglob('*.xml')]
        for filepath in tqdm(files_path):
            try:
                DOMTree = minidom.parse(str(filepath))
            except ExpatError:
                pass
            
            sentences_elements = DOMTree.getElementsByTagName('sentence')
            for sentence_dom in sentences_elements:
                entity_dict = self.get_entity_dict(sentence_dom)

                pairs = sentence_dom.getElementsByTagName('pair')
                sentence_text = sentence_dom.getAttribute('text')
                for pair in pairs:
                    relation = pair.getAttribute('type')
                    if not relation:
                        continue
                    e1_id = pair.getAttribute('e1')
                    e2_id = pair.getAttribute('e2')
                    
                    other_entities = self.get_other_entities(entity_dict, e1_id, e2_id)
                    
                    e1_data = entity_dict[e1_id]
                    e2_data = entity_dict[e2_id]
                    
                    tagged_sentence = self.tag_sentence(sentence_text, e1_data, e2_data, other_entities)
                    
                    yield tagged_sentence, relation
                        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_filepath', default='benchmark/raw_ddi/DDICorpus/Train/', 
        help='Input path of training examples')
    parser.add_argument('--test_filepath', default='benchmark/raw_ddi/DDICorpus/Test/', 
        help='Input path of test examples')
    parser.add_argument('--nlp_tool', default='spacy', choices=config.NLP_TOOLS,
        help='NLP tool name')
    parser.add_argument('--nlp_model', default='en_core_web_sm',
        help='NLP tool model name')

    args = parser.parse_args()
    
    converter = DDIConverter(args.nlp_tool, args.nlp_model)
    converter.create_dataset(converter.get_sentences(args.train_filepath), converter.get_sentences(args.test_filepath))
