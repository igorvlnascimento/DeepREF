import spacy
from spacy.cli import download as spacy_download

from deepref.nlp.nlp_tool import NLPTool, ParsedToken, Sentence, Token

class SpacyNLPTool(NLPTool):
    def __init__(self, model:str = None):
        super().__init__(model)
        self.model = model if model is not None else 'en_core_web_trf'
        if self.model not in spacy.util.get_installed_models():
            spacy_download(self.model)
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

    def parse_for_sdp(self, sentence: str) -> list[ParsedToken]:
        """Parse *sentence* with spaCy and return a list of :class:`ParsedToken` objects."""
        doc = self.nlp(sentence)
        return [
            ParsedToken(
                idx=token.i,
                text=token.text,
                dep_=token.dep_,
                head_idx=token.head.i,
                char_start=token.idx,
                char_end=token.idx + len(token.text),
            )
            for token in doc
        ]
    
    def parse_to_sentence(self,
                         text: str,
                         subj_text: str,
                         obj_text: str,
                         relation: str = "unknown") -> Sentence:
        """
        Parse a raw sentence with spaCy and return a Sentence object.

        Usage:
            import spacy
            # nlp = spacy.load("en_core_web_sm")   # loaded once externally
            sentence = parse_with_spacy(
                "He was not a relative of Mike Cane",
                subj_text="He",
                obj_text="Mike Cane",
                relation="no_relation"
            )
        """
        try:
            doc = self.nlp(text)
            tokens = [Token(i=t.i, text=t.text, dep_=t.dep_,
                            head_i=t.head.i, pos_=t.pos_)
                    for t in doc]

            # Locate entity spans by text search
            words = [t.text for t in doc]
            subj_words = subj_text.split()
            obj_words  = obj_text.split()

            def find_span(target_words):
                for start in range(len(words)):
                    if words[start:start+len(target_words)] == target_words:
                        return (start, start + len(target_words) - 1)
                raise ValueError(f"Entity '{target_words}' not found in sentence.")

            subj_span = find_span(subj_words)
            obj_span  = find_span(obj_words)

            return Sentence(tokens=tokens, subj_span=subj_span,
                            obj_span=obj_span, relation=relation)

        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found.\n"
                "Install it with:  python -m spacy download en_core_web_sm\n"
                "Falling back to manually defined example sentences."
            )

    def get_words(self, sentence: str):
        doc = self.nlp(sentence)
        return [token for token in doc]
    
    def get_entity_head(self, word: spacy.tokens.token.Token) -> int:
        return word.head.i
    
    def get_deprel(self, word: spacy.tokens.token.Token) -> str:
        return word.dep_
    
    def get_pos(self, word: spacy.tokens.token.Token) -> str:
        return word.pos_