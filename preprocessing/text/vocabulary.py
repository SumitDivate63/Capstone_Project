from collections import Counter
from typing import List, Dict
import joblib
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)


class Vocabulary:
    """
    Lightweight, deterministic vocabulary.
    Maps tokens to IDs natively mapping out <PAD>, <UNK>, <BOS>, <EOS>.
    """
    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        self.is_fitted = False

    def build_vocabulary(self, tokenized_texts: List[List[str]]) -> None:
        """Constructs ID mappings exclusively on training distributions."""
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)
        
        self.word2id = {token: idx for idx, token in enumerate(self.special_tokens)}
        idx = len(self.special_tokens)
        
        allowed_space = self.vocab_size - len(self.special_tokens)
        for word, count in counter.most_common(allowed_space):
            if word not in self.word2id:
                self.word2id[word] = idx
                idx += 1
                
        self.id2word = {i: w for w, i in self.word2id.items()}
        self.is_fitted = True
        logger.info(f"Built native vocabulary. Size: {len(self.word2id)}")

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        if not self.is_fitted:
            raise RuntimeError("Vocabulary has not mapped to any text data. Run fit() first.")
        unk_id = self.word2id[self.unk_token]
        return [self.word2id.get(token, unk_id) for token in tokens]

    def save(self, path: Path) -> None:
        """Persists internal states securely to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"word2id": self.word2id, "id2word": self.id2word}, path)
        logger.info(f"Saved Text Vocabulary state natively to {path}")

    def load(self, path: Path) -> None:
        """Reloads stored state bindings."""
        if not path.exists():
            raise FileNotFoundError(f"Missing valid Vocabulary state at {path}")
        data = joblib.load(path)
        self.word2id = data["word2id"]
        self.id2word = data["id2word"]
        self.is_fitted = True
        logger.info(f"Loaded valid Text Vocabulary. Size: {len(self.word2id)}")
