from __future__ import annotations

from typing import Optional

from deepref.dataset.dataset import Dataset
from deepref.dataset.sentence import Sentence
from deepref.dataset.preprocessors.preprocessor import Preprocessor


class DigitBlindingPreprocessor(Preprocessor):
    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        preprocessing_types: Optional[list] = None,
        entity_replacement: Optional[str] = None,
    ):
        super().__init__(dataset, preprocessing_types, entity_replacement)

    def preprocess_dataset(self) -> Dataset:
        return self._apply_to_all(self.digit_blinding)

    def digit_blinding(self, sentence: Sentence) -> Sentence:
        for i, pos in enumerate(sentence.pos_tags):
            if pos == 'NUM':
                sentence.original_sentence[i] = 'DIGIT'
        return sentence
