import pandas as pd

from deepref.dataset.example_generator import ExampleGenerator

class DatasetPreprocessor():
    def remove_whitespace(self, line: str) -> str:
        return " ".join(line.split()).strip()

    def parse_position(self, position: str) -> tuple:
        start, end = position.split('-')
        return int(start), int(end)

    def sort_position_keys(self, position_dict: dict) -> list:
        return sorted(position_dict, key=lambda x: int(x.split('-')[0]))

    def create_positions_dict(self, e1, e2, other_entities) -> dict:
        position_dict = {}
        for pos in e1['charOffset']:
            position_dict.setdefault(pos, {'start': 'ENTITYSTART', 'end': 'ENTITYEND'})
        for pos in e2['charOffset']:
            position_dict.setdefault(pos, {'start': 'ENTITYOTHERSTART', 'end': 'ENTITYOTHEREND'})
        for other_ent in other_entities:
            for pos in other_ent['charOffset']:
                position_dict.setdefault(pos, {'start': 'ENTITYUNRELATEDSTART', 'end': 'ENTITYUNRELATEDEND'})
        return position_dict

    def get_other_entities(self, entity_dict: dict, e1, e2) -> list:
        excluded = {e1, e2}
        return [v for k, v in entity_dict.items() if k not in excluded]

    def tag_sentence(self, sentence: str, e1_data, e2_data, other_entities) -> str:
        position_dict = self.create_positions_dict(e1_data, e2_data, other_entities)
        sorted_positions = self.sort_position_keys(position_dict)
        tagged_sentence = ''
        prev_end = 0
        for i, curr_pos in enumerate(sorted_positions):
            curr_start, curr_end = self.parse_position(curr_pos)
            tags = position_dict[curr_pos]
            prefix = sentence[:curr_start] if i == 0 else (sentence[prev_end + 1:curr_start] or ' ')
            tagged_sentence += f"{prefix} {tags['start']} {sentence[curr_start:curr_end + 1]} {tags['end']} "
            prev_end = curr_end
        if sorted_positions and prev_end < len(sentence) - 1:
            tagged_sentence += ' ' + sentence[prev_end + 1:]
        return self.remove_whitespace(tagged_sentence)

    def get_entity_dict(self, *args):
        raise NotImplementedError

    def get_sentences(self, path):
        raise NotImplementedError
    
    def write_dataframe(self, dataset_name, sentences, nlp_tool):

        generator = ExampleGenerator(nlp_tool=nlp_tool)
        processed_sentences = [generator.generate(tagged_sentence, relation) for tagged_sentence, relation in sentences]
        columns = 'original_sentence,e1,e2,relation_type,pos_tags,dependencies_labels,ner,sk_entities'.split(',')
        data = [
            {'original_sentence': tokens, 'e1': entity1, 'e2': entity2, 'relation_type': relation_type,
             'pos_tags': pos_tags, 'dependencies_labels': dependencies_labels, 'ner': ner, 'sk_entities': sk_entities}
            for tokens, pos_tags, dependencies_labels, ner, _, entity1, entity2, relation_type, sk_entities in processed_sentences
        ]
        df = pd.DataFrame(data, columns=columns)
        output_path = f"benchmark/{dataset_name}.csv"
        df.to_csv(output_path, sep='\t', encoding='utf-8', index=False)
        print(f"Written {len(df)} rows to {output_path}")
