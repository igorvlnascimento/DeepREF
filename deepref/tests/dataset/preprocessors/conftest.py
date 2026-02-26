import pytest

from deepref.dataset.sentence import Sentence


def make_sentence(
    original_sentence: str,
    entity1: str,
    entity2: str,
    pos_tags: str,
    ner: str,
    dependencies_labels: str = "det nsubj nsubj ROOT compound dobj dobj",
    relation_type: str = "per:city_of_birth",
) -> Sentence:
    sentence = Sentence("", "", nlp_tool=None)
    sentence.load_sentence(
        original_sentence=original_sentence,
        entity1=entity1,
        entity2=entity2,
        relation_type=relation_type,
        pos_tags=pos_tags,
        dependencies_labels=dependencies_labels,
        ner=ner,
        sk_entities="{}",
    )
    return sentence


@pytest.fixture
def base_sentence() -> Sentence:
    """
    Tokens: ["the", "john", "smith", "visited", "new", "york", "yesterday"]
    Indices:   0       1       2         3         4      5         6
    entity1: "john smith" -> [1, 3]
    entity2: "new york"   -> [4, 6]
    """
    return make_sentence(
        original_sentence="the john smith visited new york yesterday",
        entity1="{'name': 'john smith', 'position': [1, 3]}",
        entity2="{'name': 'new york', 'position': [4, 6]}",
        pos_tags="DT NNP NNP VBD NNP NNP NN",
        ner="O PER PER O LOC LOC O",
    )
