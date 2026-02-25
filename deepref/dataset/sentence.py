from deepref.nlp.nlp_tool import NLPTool
from deepref.nlp.semantic_knowledge import get_semantic

from ast import literal_eval


class Sentence():
    def __init__(self, tagged_sentence: str, relation_type: str, nlp_tool: NLPTool = None):
        self.nlp_tool = nlp_tool
        self.tokens, self.pos_tags, self.dependencies_labels, self.ner = (
            self.nlp_tool.parse(tagged_sentence) if self.nlp_tool is not None else (None, None, None, None)
        )
        self.original_sentence = (
            self.nlp_tool.untag_sentence(" ".join(self.tokens)).split() if self.nlp_tool is not None else None
        )
        self.entity1, self.entity2 = self.get_entities() if self.nlp_tool is not None else ({}, {})
        self.relation_type = relation_type
        self.sk_entities = (
            get_semantic().extract([self.entity1["name"], self.entity2["name"]]) if self.nlp_tool is not None else {}
        )

    def get_entities(self):
        tokens = self.tokens
        while "ENTITYUNRELATEDSTART" in tokens:
            tokens.remove("ENTITYUNRELATEDSTART")
        while "ENTITYUNRELATEDEND" in tokens:
            tokens.remove("ENTITYUNRELATEDEND")
        if tokens.index("ENTITYEND") < tokens.index("ENTITYOTHEREND"):
            pos1 = [tokens.index("ENTITYSTART"), tokens.index("ENTITYEND") - 1]
            pos2 = [tokens.index("ENTITYOTHERSTART") - 2, tokens.index("ENTITYOTHEREND") - 3]
        else:
            pos1 = [tokens.index("ENTITYSTART") - 2, tokens.index("ENTITYEND") - 3]
            pos2 = [tokens.index("ENTITYOTHERSTART"), tokens.index("ENTITYOTHEREND") - 1]
        e1_name = " ".join(self.original_sentence[pos1[0]:pos1[1]])
        e2_name = " ".join(self.original_sentence[pos2[0]:pos2[1]])
        return {"name": e1_name.lower(), "position": pos1}, {"name": e2_name.lower(), "position": pos2}

    def get_sentence_info(self):
        return [
            " ".join(self.original_sentence).lower(),
            self.entity1,
            self.entity2,
            self.relation_type,
            " ".join(self.pos_tags),
            " ".join(self.dependencies_labels),
            " ".join(self.ner),
            self.sk_entities,
        ]

    def load_sentence(self, original_sentence, entity1, entity2, relation_type, pos_tags, dependencies_labels, ner, sk_entities):
        self.original_sentence = original_sentence.split()
        self.pos_tags = pos_tags.split()
        self.dependencies_labels = dependencies_labels.split()
        self.ner = ner.split()
        self.entity1 = literal_eval(entity1)
        self.entity2 = literal_eval(entity2)
        self.relation_type = relation_type
        self.sk_entities = literal_eval(sk_entities)