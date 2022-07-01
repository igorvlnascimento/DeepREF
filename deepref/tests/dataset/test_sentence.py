from deepref.utils.spacy_nlp_tool import SpacyNLPTool
from deepref.dataset.sentence import Sentence

def test_should_return_correct_info_on_sentence_class():
    tagged_sentence = "this ENTITYSTART outline ENTITYEND focuses on ENTITYOTHERSTART spirituality ENTITYOTHEREND , esotericism, mysticism, religion and/or parapsychology."
    relation = 'Message-Topic(e1,e2)'
    nlp_tool = SpacyNLPTool()
    s = Sentence(tagged_sentence, relation, nlp_tool)
    assert s.tokens == "this ENTITYSTART outline ENTITYEND focuses on ENTITYOTHERSTART spirituality ENTITYOTHEREND , esotericism , mysticism , religion and/or parapsychology .".split()
    assert s.dependencies_labels == "det nsubj root prep pobj punct conj punct conj punct conj cc conj punct".split()
    assert s.pos_tags == "DET NOUN VERB ADP NOUN PUNCT NOUN PUNCT NOUN PUNCT NOUN CCONJ NOUN PUNCT".split()
    assert s.sk_entities == {'ses1': ['boundary', 'extremity'], 'ses2': ['property', 'possession']}
    assert s.entity1 == {'name': 'outline', 'position': 1}
    assert s.entity2 == {'name': 'spirituality', 'position': 4}
    