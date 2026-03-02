import re
from abc import ABC, abstractmethod
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from deepref.dataset.example_generator import ExampleGenerator

class DatasetPreprocessor(ABC):
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

    @abstractmethod
    def get_entity_dict(self, *args):
        ...

    @abstractmethod
    def get_sentences(self, path):
        ...

    def remove_entity_marks(self, sentence: str) -> str:
        """Remove all ENTITY* marker tokens from a tagged sentence."""
        pattern = (
            r'\b(ENTITYSTART|ENTITYEND'
            r'|ENTITYOTHERSTART|ENTITYOTHEREND'
            r'|ENTITYUNRELATEDSTART|ENTITYUNRELATEDEND)\b'
        )
        return self.remove_whitespace(re.sub(pattern, '', sentence))

    def _build_df(self, sentences, nlp_tool) -> pd.DataFrame:
        out = defaultdict(list)
        example_generator = ExampleGenerator(nlp_tool=nlp_tool)
        for tagged_sentence, relation in tqdm(sentences):
            example_dict = example_generator.generate(tagged_sentence, relation)
            for k, v in example_dict.items():
                out[k].append(" ".join(v) if isinstance(v, list) else v)
        return pd.DataFrame(out)

    def write_csv(self, dataset_name, sentences, nlp_tool):
        df = self._build_df(sentences, nlp_tool)
        output_path = f"benchmark/{dataset_name}.csv"
        df.to_csv(output_path, sep='\t', encoding='utf-8', index=False)
        print(f"Written {len(df)} rows to {output_path}")

    def write_split_csvs(self, dataset_name, train_sentences, test_sentences, nlp_tool):
        """Write separate train and test CSV files using the dataset's natural split.

        Outputs:
            benchmark/{dataset_name}_train.csv
            benchmark/{dataset_name}_test.csv
        """
        print(f"Processing train split for '{dataset_name}' …")
        train_df = self._build_df(train_sentences, nlp_tool)
        print(f"Processing test split for '{dataset_name}' …")
        test_df = self._build_df(test_sentences, nlp_tool)

        train_path = f"benchmark/{dataset_name}_train.csv"
        test_path = f"benchmark/{dataset_name}_test.csv"

        train_df.to_csv(train_path, sep='\t', encoding='utf-8', index=False)
        test_df.to_csv(test_path, sep='\t', encoding='utf-8', index=False)

        print(f"Written {len(train_df)} rows to {train_path}")
        print(f"Written {len(test_df)} rows to {test_path}")
