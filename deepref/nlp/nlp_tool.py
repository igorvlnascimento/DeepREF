class NLPTool():
    def __init__(self, model:str):
        self.model = model
    
    def parse(self, sentence: str) -> tuple:
        """ Parsing sentences """
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