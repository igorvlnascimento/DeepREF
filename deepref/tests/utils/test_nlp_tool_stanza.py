from deepref.utils.stanza_nlp_tool import StanzaNLPTool

def test_should_return_correct_parsing():
    sentence = 'the most common audits were about waste and recycling.'
    tokens = ['the', 'most', 'common', 'audits', 'were', 'about', 'waste', 'and', 'recycling', '.']
    upos = ['DET', 'ADV', 'ADJ', 'NOUN', 'AUX', 'ADP', 'NOUN', 'CCONJ', 'NOUN', 'PUNCT']
    deps = ['det', 'advmod', 'amod', 'nsubj', 'cop', 'case', 'root', 'cc', 'conj', 'punct']
    ner = ['O','O','O','O','O','O','O','O','O','O']
    nlp_tool = StanzaNLPTool()
    assert nlp_tool.parse(sentence) == (tokens, upos, deps, ner)
    