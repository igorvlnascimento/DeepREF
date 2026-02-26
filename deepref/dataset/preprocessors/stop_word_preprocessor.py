from __future__ import annotations

from typing import Optional

import nltk
from nltk.corpus import stopwords

from deepref.dataset.dataset import Dataset
from deepref.dataset.sentence import Sentence
from deepref.dataset.preprocessors.preprocessor import Preprocessor


class StopWordPreprocessor(Preprocessor):
    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        preprocessing_types: Optional[list] = None,
        entity_replacement: Optional[str] = None,
    ):
        super().__init__(dataset, preprocessing_types, entity_replacement)
        nltk.download('stopwords', quiet=True)
        self._stop_words = set(stopwords.words('english')) - {'o'}

    def preprocess_dataset(self) -> Dataset:
        return self._apply_to_all(self.stop_words_removal)

    def stop_words_removal(self, sentence: Sentence) -> Sentence:
        entity_indexes = self._entity_indexes(sentence)
        indexes = [
            j for j, token in enumerate(sentence.original_sentence)
            if token in self._stop_words and j not in entity_indexes
        ]
        return self.process_sentence(sentence, indexes)
