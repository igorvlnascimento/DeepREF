import os
import stanza
from stanza.pipeline.core import DownloadMethod
from deepref.nlp.nlp_tool import NLPTool


class StanzaNLPTool(NLPTool):
    def __init__(self, model=None, lang="en"):
        super().__init__(model)
        self.lang = lang
        self.model = model if model is not None else "default"

        self.resources_dir = os.getenv("STANZA_RESOURCES_DIR", None)
        self.processors = "tokenize,mwt,pos,lemma,depparse,ner"

        try:
            self.nlp = stanza.Pipeline(
                lang=self.lang,
                dir=self.resources_dir,
                processors=self.processors,
                tokenize_no_ssplit=True,
                download_method=DownloadMethod.REUSE_RESOURCES,
            )
        except Exception:
            stanza.download(
                self.lang,
                package=self.model,
                processors=self.processors,
                dir=self.resources_dir,
            )
            self.nlp = stanza.Pipeline(
                lang=self.lang,
                dir=self.resources_dir,
                processors=self.processors,
                tokenize_no_ssplit=True,
                download_method=DownloadMethod.REUSE_RESOURCES,
            )

    @staticmethod
    def _norm_ner(tag: str) -> str:
        if not tag or tag == "O":
            return "O"
        if "-" in tag:
            return tag.split("-", 1)[1]
        return tag

    def parse(self, tagged_sentence: str):
        tokens = tagged_sentence.split()
        sentence = self.untag_sentence(tagged_sentence)

        doc = self.nlp(sentence)

        words = [w for sent in doc.sentences for w in sent.words]
        upos = [w.upos for w in words]
        deps = [(w.deprel or "dep").lower() for w in words]

        ner = []
        for sent in doc.sentences:
            for tok in sent.tokens:
                tag = self._norm_ner(tok.ner)
                for _ in tok.words:
                    ner.append(tag)

        assert len(words) == len(upos) == len(deps) == len(ner)
        return tokens, upos, deps, ner