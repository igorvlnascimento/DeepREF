import stanza
from deepref.nlp.nlp_tool import NLPTool

class StanzaNLPTool(NLPTool):
    def __init__(self, model=None):
        super().__init__(model)
        self.model = model if model is not None else 'default'
        stanza.download('en', package=self.model, processors='tokenize,ner')
        self.nlp = stanza.Pipeline(lang='en', processors="tokenize,ner,depparse,pos,lemma", tokenize_no_ssplit=True)

    def parse(self, sentence):
        doc = self.nlp(sentence)
        tokens = [token.text for sent in doc.sentences for token in sent.words]
        upos = [token.upos for sent in doc.sentences for token in sent.words]
        deps = [token.deprel for sent in doc.sentences for token in sent.words]
        ner = [token.ner for sent in doc.sentences for token in sent.tokens]
        
        return tokens, upos, deps, ner