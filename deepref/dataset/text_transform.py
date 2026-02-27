from abc import ABC, abstractmethod
import string

import nltk
from nltk.corpus import stopwords

class TextTransformer(ABC):
    @abstractmethod
    def transform(self, text: str) -> str:
        ...

class PuctuationRemover(TextTransformer):
    def transform(self, text: str) -> str:
        tokens = text.split()
        tokens = [t for t in tokens if t not in string.punctuation]
        return " ".join(tokens)

class StopwordRemover(TextTransformer):
    def __init__(self):
        nltk.download('stopwords', quiet=True)
        self._stop_words = set(stopwords.words('english')) - {'o'}

    def transform(self, text: str) -> str:
        tokens = text.split()
        tokens = [t for t in tokens if t.lower() not in self._stop_words]
        return " ".join(tokens)
    
class DigitBlinding(TextTransformer):
    def transform(self, text: str) -> str:
        tokens = text.split()
        tokens = ["DIGIT" if t.isdigit() else t for t in tokens]
        return " ".join(tokens)
    
class EntityBlinding(TextTransformer):
    def __init__(self, pos1, pos2, text_e1, text_e2):
        self._pos1 = pos1
        self._pos2 = pos2
        self._text_e1 = text_e1
        self._text_e2 = text_e2

    def transform(self, text: str) -> str:
        tokens = text.split()
        tokens[self._pos1] = self._text_e1
        tokens[self._pos2] = self._text_e2
        return " ".join(tokens)
    
class BracketsOrParenthesisRemover(TextTransformer):
    def transform(self, text: str) -> str:
        tokens = text.split()
        tokens = [t for t in tokens if t not in ["(", ")", "[", "]"]]
        return " ".join(tokens)