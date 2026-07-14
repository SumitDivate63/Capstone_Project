import re
from typing import List

class WhitespaceTokenizer:
    """Simple tokenizer splitting on whitespaces natively."""
    def fit(self, texts: List[str]) -> None:
        pass

    def transform(self, text: str) -> List[str]:
        return text.split()

    def fit_transform(self, texts: List[str]) -> List[List[str]]:
        return [self.transform(t) for t in texts]

class SimpleRegexTokenizer:
    """Model agnostic regex extraction splitting tokens vs punctuation."""
    def __init__(self, pattern: str = r"\w+|[^\w\s]"):
        self.pattern = re.compile(pattern)

    def fit(self, texts: List[str]) -> None:
        pass

    def transform(self, text: str) -> List[str]:
        return self.pattern.findall(text)

    def fit_transform(self, texts: List[str]) -> List[List[str]]:
        return [self.transform(t) for t in texts]
