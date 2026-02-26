from __future__ import annotations

from typing import Optional

from deepref.dataset.dataset import Dataset
from deepref.dataset.sentence import Sentence
from deepref.dataset.preprocessors.preprocessor import Preprocessor


class EntityBlindingPreprocessor(Preprocessor):
    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        preprocessing_types: Optional[list] = None,
        entity_replacement: Optional[str] = None,
        type: str = 'ner',
    ):
        self.type = type
        super().__init__(dataset, preprocessing_types, entity_replacement)

    def preprocess_dataset(self) -> Dataset:
        return self._apply_to_all(self.entity_blinding)

    def entity_blinding(self, sentence: Sentence) -> Sentence:
        position1 = sentence.entity1['position']
        position2 = sentence.entity2['position']

        if self.type == 'ner':
            replacement = [sentence.ner[position1[0]], sentence.ner[position2[0]]]
        else:
            replacement = [self.entity_replacement, self.entity_replacement]

        if position1[0] < position2[0]:
            entity1_span = position1[-1] - position1[0]
            sentence.original_sentence = (
                sentence.original_sentence[:position1[0]] + [replacement[0]] +
                sentence.original_sentence[position1[-1]:position2[0]] + [replacement[1]] +
                sentence.original_sentence[position2[-1]:]
            )
            sentence.entity1['position'] = [position1[0], position1[0] + 1]
            position2[0] -= entity1_span - 1
            sentence.entity2['position'] = [position2[0], position2[0] + 1]
        else:
            entity2_span = position2[-1] - position2[0]
            sentence.original_sentence = (
                sentence.original_sentence[:position2[0]] + [replacement[1]] +
                sentence.original_sentence[position2[-1]:position1[0]] + [replacement[0]] +
                sentence.original_sentence[position1[-1]:]
            )
            sentence.entity2['position'] = [position2[0], position2[0] + 1]
            position1[0] -= entity2_span - 1
            sentence.entity1['position'] = [position1[0], position1[0] + 1]

        assert sentence.original_sentence[sentence.entity1['position'][0]] == replacement[0]
        assert sentence.original_sentence[sentence.entity2['position'][0]] == replacement[1]
        return sentence
