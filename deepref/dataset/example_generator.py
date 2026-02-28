from deepref.nlp.nlp_tool import NLPTool
from deepref.nlp.semantic_knowledge import SemanticKNWL

class ExampleGenerator:
    def __init__(self, nlp_tool: NLPTool):
        self.nlp_tool = nlp_tool

    def generate(self, tagged_sentence, relation_type):
        tokens, pos_tags, dependencies_labels, ner = self.nlp_tool.parse(tagged_sentence)
        original_sentence = self.nlp_tool.untag_sentence(" ".join(tokens)).split()
        entity1, entity2 = self._get_entities(tokens, original_sentence)
        sk_entities = SemanticKNWL().extract([entity1['name'], entity2['name']])
        return {'original_sentence': original_sentence,
                'e1': entity1,
                'e2': entity2,
                'relation_type': relation_type,
                'pos_tags': pos_tags,
                'dependencies_labels': dependencies_labels,
                'ner': ner,
                'sk_entities': sk_entities}

    def _get_entities(self, tokens, original_sentence):
        tokens = [t for t in tokens if t not in ("ENTITYUNRELATEDSTART", "ENTITYUNRELATEDEND")]
        if tokens.index("ENTITYEND") < tokens.index("ENTITYOTHEREND"):
            pos1 = [tokens.index("ENTITYSTART"), tokens.index("ENTITYEND") - 1]
            pos2 = [tokens.index("ENTITYOTHERSTART") - 2, tokens.index("ENTITYOTHEREND") - 3]
        else:
            pos1 = [tokens.index("ENTITYSTART") - 2, tokens.index("ENTITYEND") - 3]
            pos2 = [tokens.index("ENTITYOTHERSTART"), tokens.index("ENTITYOTHEREND") - 1]
        e1_name = " ".join(original_sentence[pos1[0]:pos1[1]])
        e2_name = " ".join(original_sentence[pos2[0]:pos2[1]])
        return {'name': e1_name.lower(), 'position': pos1}, {'name': e2_name.lower(), 'position': pos2}
