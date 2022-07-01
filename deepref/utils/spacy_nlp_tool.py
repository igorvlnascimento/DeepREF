import spacy
import subprocess
from deepref.utils.nlp_tool import NLPTool

class SpacyNLPTool(NLPTool):
    def __init__(self, model:str = None):
        super().__init__(model)
        self.model = model if model is not None else 'en_core_web_sm'
        if self.model not in spacy.util.get_installed_models():
            subprocess.call(["python", "-m", "spacy", "download", self.model])
        self.nlp = spacy.load(self.model)
    
    def parse(self, tagged_sentence):
        doc_tag = self.nlp(tagged_sentence)
        tokens = [token.text for token in doc_tag]
        sentence = self.untag_sentence(tagged_sentence)
        doc = self.nlp(sentence)
        upos = [token.pos_ for token in doc]
        deps = [token.dep_.lower() for token in doc]
        ner = ["O"] * len([token.text for token in doc])
        for ent in doc.ents:
            for i in range(ent.start, ent.end):
                ner[i] = ent.label_
        assert len([token.text for token in doc]) == len(upos) == len(deps) == len(ner)
                
        return tokens, upos, deps, ner