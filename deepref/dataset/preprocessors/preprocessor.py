from __future__ import annotations

from typing import Optional

from tqdm import tqdm

from deepref.dataset.dataset import Dataset
from deepref.dataset.sentence import Sentence


class Preprocessor:
    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        preprocessing_types: Optional[list] = None,
        entity_replacement: Optional[str] = None,
    ):
        self.dataset = dataset
        self.preprocessing_types = preprocessing_types
        self.entity_replacement = entity_replacement

    def preprocess_dataset(self) -> Dataset:
        raise NotImplementedError

    def _apply_to_all(self, transform) -> Dataset:
        assert self.dataset is not None
        for split in [self.dataset.train_sentences, self.dataset.test_sentences]:
            for i, sentence in tqdm(enumerate(split)):
                split[i] = transform(sentence)
        return self.dataset

    def _entity_indexes(self, sentence: Sentence) -> set:
        return set(range(*sentence.entity1['position'])) | set(range(*sentence.entity2['position']))

    def process_sentence(self, sentence: Sentence, indexes: list) -> Sentence:
        indexes_set = set(indexes)

        def count_before(threshold):
            return sum(1 for i in indexes_set if i < threshold)

        def filter_sequence(seq):
            return [item for i, item in enumerate(seq) if i not in indexes_set]

        ent1_start = sentence.entity1['position'][0]
        ent2_start = sentence.entity2['position'][0]

        offset1 = count_before(ent1_start)
        offset2 = count_before(ent2_start)

        sentence.original_sentence = filter_sequence(sentence.original_sentence)
        sentence.pos_tags = filter_sequence(sentence.pos_tags)
        sentence.dependencies_labels = filter_sequence(sentence.dependencies_labels)
        sentence.ner = filter_sequence(sentence.ner)

        sentence.entity1['position'][0] -= offset1
        sentence.entity1['position'][1] -= offset1
        sentence.entity2['position'][0] -= offset2
        sentence.entity2['position'][1] -= offset2

        assert " ".join(sentence.original_sentence[sentence.entity1['position'][0]:sentence.entity1['position'][1]]) == sentence.entity1['name']
        assert " ".join(sentence.original_sentence[sentence.entity2['position'][0]:sentence.entity2['position'][1]]) == sentence.entity2['name']
        assert len(sentence.original_sentence) == len(sentence.pos_tags) == len(sentence.dependencies_labels) == len(sentence.ner)

        return sentence
