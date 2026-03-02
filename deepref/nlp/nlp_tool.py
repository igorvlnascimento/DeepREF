from dataclasses import dataclass


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
