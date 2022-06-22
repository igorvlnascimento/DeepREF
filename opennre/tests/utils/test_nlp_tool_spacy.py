from opennre.utils.spacy_nlp_tool import SpacyNLPTool

def test_should_return_correct_parsing():
    sentence = 'the most common audits were about waste and recycling.'
    tokens = ['the', 'most', 'common', 'audits', 'were', 'about', 'waste', 'and', 'recycling', '.']
    upos = ['DET', 'ADV', 'ADJ', 'NOUN', 'AUX', 'ADP', 'NOUN', 'CCONJ', 'NOUN', 'PUNCT']
    deps = ['det', 'advmod', 'amod', 'nsubj', 'root', 'prep', 'pobj', 'cc', 'conj', 'punct']
    ner = ['O','O','O','O','O','O','O','O','O','O']
    nlp_tool = SpacyNLPTool('en_core_web_sm')
    assert nlp_tool.parse(sentence) == (tokens, upos, deps, ner)
    