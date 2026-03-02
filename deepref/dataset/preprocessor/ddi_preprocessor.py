import argparse
from pyexpat import ExpatError
from xml.dom import minidom
from pathlib import Path
from tqdm import tqdm

from deepref.dataset.preprocessor.dataset_preprocessor import DatasetPreprocessor
from deepref.nlp.spacy_nlp_tool import SpacyNLPTool

class DDIPreprocessor(DatasetPreprocessor):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CSV from raw DDI XML files")
    parser.add_argument("--path", required=True, help="Path to directory containing DDI XML files")
    args = parser.parse_args()

    tool = SpacyNLPTool("en_core_web_trf")
    preprocessor = DDIPreprocessor()

    base = Path(args.path)
    train_path = next((p for p in base.rglob("Train") if p.is_dir()), None)
    test_path = next((p for p in base.rglob("Test") if p.is_dir()), None)

    if train_path and test_path:
        train_sentences = list(preprocessor.get_sentences(str(train_path)))
        test_sentences = list(preprocessor.get_sentences(str(test_path)))
        preprocessor.write_split_csvs("ddi", train_sentences, test_sentences, tool)
    else:
        preprocessor.write_csv("ddi", preprocessor.get_sentences(args.path), tool)
