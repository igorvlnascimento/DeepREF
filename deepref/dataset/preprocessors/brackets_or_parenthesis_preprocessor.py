from __future__ import annotations

from typing import Optional

from deepref.dataset.dataset import Dataset
from deepref.dataset.sentence import Sentence
from deepref.dataset.preprocessors.preprocessor import Preprocessor


class BracketsPreprocessor(Preprocessor):
    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        preprocessing_types: Optional[list] = None,
        entity_replacement: Optional[str] = None,
    ):
        super().__init__(dataset, preprocessing_types, entity_replacement)

    def preprocess_dataset(self) -> Dataset:
        return self._apply_to_all(self.remove_brackets_or_parenthesis)

    def remove_brackets_or_parenthesis(self, sentence: Sentence) -> Sentence:
        entity_indexes = self._entity_indexes(sentence)
        indexes = []
        inside_brackets = False

        for j, token in enumerate(sentence.original_sentence):
            if j in entity_indexes:
                continue
            if token in ('(', '['):
                inside_brackets = True
                indexes.append(j)
            elif token in (')', ']'):
                inside_brackets = False
                indexes.append(j)
            elif inside_brackets:
                indexes.append(j)

        return self.process_sentence(sentence, indexes)
