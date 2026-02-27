import os

import pandas as pd

from deepref.nlp.nlp_tool import NLPTool

from deepref.dataset.dataset import Dataset

class DatasetPreprocessor():

    def __init__(self, dataset: Dataset, nlp_tool: NLPTool = None):
        self.dataset = dataset
        self.nlp_tool = nlp_tool

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
    
    def write_dataframe(self, dataset_name, sentences):
        data = []
        #val_data = []
        test_data = []
        for sentence in sentences:
            data.append(sentence.get_sentence_info())
        #for sentence in self.val_sentences:
        #    val_data.append(sentence.get_sentence_info())
        columns = 'original_sentence,e1,e2,relation_type,pos_tags,dependencies_labels,ner,sk_entities'.split(',')
        df = pd.DataFrame(data,
            columns=columns)
        #val_df = pd.DataFrame(val_data,
        #    columns=columns)
        test_df = pd.DataFrame(test_data,
            columns=columns)
        df.to_csv(f"benchmark/{dataset_name}/{dataset_name}.csv", sep='\t', encoding='utf-8', index=False)
        #val_df.to_csv(f"benchmark/{dataset_name}/original/{dataset_name}_val_original.csv", sep='\t', encoding='utf-8', index=False)
        #test_df.to_csv(f"benchmark/{dataset_name}/original/{dataset_name}_test_original.csv", sep='\t', encoding='utf-8', index=False)
