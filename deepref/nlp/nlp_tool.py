class NLPTool():
    def __init__(self, model:str):
        self.model = model
    
    def parse(self, sentence: str) -> tuple:
        """ Parsing sentences """
        pass
    
    def untag_sentence(self, tagged_sentence:str) -> str:
        tokens_copy = tagged_sentence.split()
        tokens_copy.remove("ENTITYSTART")
        tokens_copy.remove("ENTITYEND")
        tokens_copy.remove("ENTITYOTHERSTART")
        tokens_copy.remove("ENTITYOTHEREND")
        while "ENTITYUNRELATEDSTART" in tokens_copy:
            tokens_copy.remove("ENTITYUNRELATEDSTART")
        while "ENTITYUNRELATEDEND" in tokens_copy:
            tokens_copy.remove("ENTITYUNRELATEDEND")
        return " ".join(tokens_copy)