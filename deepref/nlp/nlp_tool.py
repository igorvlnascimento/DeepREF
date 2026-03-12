from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ParsedToken:
    """Tool-agnostic representation of a dependency-parsed token.

    Attributes:
        idx:        0-based position in the token list.
        text:       Surface form of the token.
        dep_:       Dependency relation label (abbreviated, e.g. ``'nsubj'``).
        head_idx:   0-based index of the syntactic head token.
                    For ROOT tokens this equals ``idx``.
        char_start: Character start offset in the sentence string.
        char_end:   Character end offset (exclusive) in the sentence string.
    """
    idx: int
    text: str
    dep_: str
    head_idx: int
    char_start: int
    char_end: int

@dataclass
class Token:
    """Lightweight token mimicking spaCy's Token interface."""
    i: int            # index in sentence
    text: str
    dep_: str         # dependency label
    head_i: int       # index of syntactic head
    pos_: str = "NOUN"

    def __repr__(self):
        return f"Token({self.i}, '{self.text}', dep='{self.dep_}', head={self.head_i})"


@dataclass
class Sentence:
    """Container for a parsed sentence + entity spans."""
    tokens: List[Token]
    subj_span: Tuple[int, int]   # (start, end) inclusive token indices
    obj_span:  Tuple[int, int]
    relation:  str = "unknown"

    @property
    def text(self):
        return " ".join(t.text for t in self.tokens)

    def span_text(self, span):
        return " ".join(self.tokens[i].text for i in range(span[0], span[1]+1))


class NLPTool():
    def __init__(self, model: str):
        self.model = model

    def parse(self, sentence: str) -> tuple:
        """Parse a (possibly tagged) sentence and return (tokens, upos, deps, ner)."""
        pass

    def parse_for_sdp(self, sentence: str) -> list[ParsedToken]:
        """Parse *sentence* and return a list of :class:`ParsedToken` objects.

        The returned list covers every token in *sentence* in order.
        Each element carries the dependency label and head index needed by
        the Shortest Dependency Path (SDP) extraction algorithm.
        """
        pass

    def parse_to_sentence(self,
                         text: str,
                         subj_text: str,
                         obj_text: str,
                         relation: str = "unknown") -> Sentence:
        """
        Parse a raw sentence and return a Sentence object.
        """
        pass

    def untag_sentence(self, tagged_sentence: str) -> str:
        tokens_copy = tagged_sentence.split()

        tags_once = [
            "ENTITYSTART", "ENTITYEND",
            "ENTITYOTHERSTART", "ENTITYOTHEREND",
        ]
        for tag in tags_once:
            while tag in tokens_copy:
                tokens_copy.remove(tag)

        for tag in ["ENTITYUNRELATEDSTART", "ENTITYUNRELATEDEND"]:
            while tag in tokens_copy:
                tokens_copy.remove(tag)

        return " ".join(tokens_copy)
    
    def get_words(self, sentence: str):
        """
        Return a list of words in the sentence.
        """
        pass

    def get_entity_head(self, word) -> int:
        """
        Return entity head index.
        """
        pass

    def get_deprel(self, word) -> str:
        """
        Return dependency relation label.
        """
        pass

    def get_pos(self, word) -> str:
        """
        Return part-of-speech tag.
        """
        pass