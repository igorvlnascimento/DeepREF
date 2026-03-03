import argparse
import json
from tqdm import tqdm
from pyexpat import ExpatError
import xml.etree.ElementTree as ET
from pathlib import Path

from deepref.dataset.preprocessor.dataset_preprocessor import DatasetPreprocessor

from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

from deepref.nlp.spacy_nlp_tool import SpacyNLPTool

class SemEval2018Preprocessor(DatasetPreprocessor):

    LABEL_MAP = {
        1: 'USAGE',
        2: 'RESULT',
        3: 'MODEL-FEATURE',
        4: 'PART_WHOLE',
        5: 'TOPIC',
        6: 'COMPARE',
    }

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
    
    def get_entity_dict_from_json(self, doc):
        """Process a JSON document, yielding (sentence, entity_dict) per sentence.

        Entities are located via char_start/char_end offsets (char_end exclusive)
        in doc['abstract'].  The yielded entity_dict uses the same charOffset
        format as get_entity_dict: {'word': str, 'charOffset': ['start-end']}
        with an inclusive end index.
        """
        punk_param = PunktParameters()
        punk_param.abbrev_types = {'e.g', 'viz', 'al'}
        tokenizer = PunktSentenceTokenizer(punk_param)

        text = doc.get('abstract', '')
        entity_dict = {}
        for entity in doc.get('entities', []):
            eid = entity['id']
            start = entity['char_start']
            end = entity['char_end']  # exclusive
            word = text[start:end]
            entity_dict[eid] = {'word': word, 'charOffset': [f'{start}-{end - 1}']}

        sentences = tokenizer.tokenize(text)
        for i, s in enumerate(sentences):
            sentences_length = (
                0 if len(' '.join(sentences[:i])) == 0
                else len(' '.join(sentences[:i])) + 1
            )
            entity_dict_sentence = {}
            for eid, edata in entity_dict.items():
                pos_start, pos_end = map(int, edata['charOffset'][0].split('-'))
                if pos_start >= sentences_length and pos_end <= len(s) + sentences_length:
                    rel_start = pos_start - sentences_length
                    rel_end = pos_end - sentences_length
                    entity_dict_sentence[eid] = {
                        'word': edata['word'],
                        'charOffset': [f'{rel_start}-{rel_end}'],
                    }
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

    def get_entity_pairs_from_json(self, relations):
        """Build an entity-pair lookup from a JSON relations list.

        Each relation is a dict with keys: label (int), arg1, arg2, reverse (bool).
        The canonical direction is arg1→arg2; when reverse=True the roles are swapped
        so that e1 is the true first argument.
        """
        entity_pairs = {}
        for rel in relations:
            label = self.LABEL_MAP.get(rel['label'], str(rel['label']))
            if rel.get('reverse', False):
                e1_id, e2_id = rel['arg2'], rel['arg1']
            else:
                e1_id, e2_id = rel['arg1'], rel['arg2']
            entity_pairs[e1_id] = {'relation': label, 'e1': e1_id, 'e2': e2_id}
        return entity_pairs

    def get_sentences(self, path):
        if list(Path(path).rglob('*.json')):
            yield from self._get_sentences_from_json(path)
        else:
            yield from self._get_sentences_from_xml(path)

    def _get_sentences_from_xml(self, path):
        for filepath in Path(path).rglob('*.xml'):
            try:
                tree = ET.parse(str(filepath))
                root = tree.getroot()
            except ExpatError:
                pass

            text_elements = root.findall('./text/')
            pairs = self.get_entity_pairs(path)

            for sentence, entity_dict in tqdm(self.get_entity_dict(text_elements)):
                for e1_id in entity_dict:
                    if e1_id in pairs:
                        e2_id = pairs[e1_id]['e2']
                        relation = pairs[e1_id]['relation'].lower()

                        other_entities = self.get_other_entities(entity_dict, e1_id, e2_id)
                        e1_data = entity_dict[e1_id]
                        if e2_id not in entity_dict:
                            continue
                        e2_data = entity_dict[e2_id]
                        tagged_sentence = self.tag_sentence(sentence, e1_data, e2_data, other_entities)

                        yield tagged_sentence, relation

    def _get_sentences_from_json(self, path):
        for filepath in sorted(Path(path).rglob('*.json')):
            content = filepath.read_text()
            stripped = content.strip()
            if stripped.startswith('['):
                docs = json.loads(stripped)
            else:
                docs = [json.loads(line) for line in stripped.splitlines() if line.strip()]

            for doc in tqdm(docs, desc=filepath.name):
                entity_pairs = self.get_entity_pairs_from_json(doc.get('relations', []))

                for sentence, entity_dict in self.get_entity_dict_from_json(doc):
                    for e1_id in entity_dict:
                        if e1_id not in entity_pairs:
                            continue
                        e2_id = entity_pairs[e1_id]['e2']
                        relation = entity_pairs[e1_id]['relation'].lower()

                        if e2_id not in entity_dict:
                            continue

                        other_entities = self.get_other_entities(entity_dict, e1_id, e2_id)
                        tagged_sentence = self.tag_sentence(
                            sentence, entity_dict[e1_id], entity_dict[e2_id], other_entities
                        )
                        yield tagged_sentence, relation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CSV from raw SemEval2018 XML files")
    parser.add_argument("--path", required=True, help="Path to directory containing SemEval2018 XML and txt files")
    args = parser.parse_args()

    if "semeval20181-1" in args.path:
        dataset_name = "semeval20181-1"
    elif "semeval20181-2" in args.path:
        dataset_name = "semeval20181-2"
    else:
        raise ValueError("Invalid path: must contain 'semeval20181-1' or 'semeval20181-2'")

    preprocessor = SemEval2018Preprocessor()
    tool = SpacyNLPTool("en_core_web_trf")

    base = Path(args.path)
    train_path = next((p for p in base.rglob("Train") if p.is_dir()), None)
    test_path = next((p for p in base.rglob("Test") if p.is_dir()), None)

    if train_path and test_path:
        train_sentences = list(preprocessor.get_sentences(str(train_path)))
        test_sentences = list(preprocessor.get_sentences(str(test_path)))
        preprocessor.write_split_csvs(dataset_name, train_sentences, test_sentences, tool)
    else:
        preprocessor.write_csv(dataset_name, preprocessor.get_sentences(args.path), tool)
