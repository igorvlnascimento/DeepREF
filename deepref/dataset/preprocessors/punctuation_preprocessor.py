from __future__ import annotations

from typing import Optional

from deepref.dataset.dataset import Dataset
from deepref.dataset.sentence import Sentence
from deepref.dataset.preprocessors.preprocessor import Preprocessor


class PunctuationPreprocessor(Preprocessor):
    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        preprocessing_types: Optional[list] = None,
        entity_replacement: Optional[str] = None,
    ):
        super().__init__(dataset, preprocessing_types, entity_replacement)

    def preprocess_dataset(self) -> Dataset:
        return self._apply_to_all(self.remove_punctuation)

    def remove_punctuation(self, sentence: Sentence) -> Sentence:
        entity_indexes = self._entity_indexes(sentence)
        indexes = [
            j for j, pos in enumerate(sentence.pos_tags)
            if pos == 'PUNCT' and j not in entity_indexes
        ]
        return self.process_sentence(sentence, indexes)
