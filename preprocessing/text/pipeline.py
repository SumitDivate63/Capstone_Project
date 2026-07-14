import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

from .preprocessor import TextPreprocessor
from .tokenizer import WhitespaceTokenizer, SimpleRegexTokenizer
from .vocabulary import Vocabulary
from .sequence import prepare_sequence
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TextPreprocessingConfig:
    max_sequence_length: int = 512
    lowercase: bool = True
    keep_ellie: bool = False
    tokenizer: str = "whitespace"
    vocab_size: int = 30000
    padding: str = "max_length"
    truncation: bool = True


class TextPreprocessingPipeline:
    """
    Absolute Pipeline Controller mapping raw Text structures from Dataset loaders purely into PyTorch primitive forms ready for Deep Learning encoding models seamlessly.
    """

    def __init__(self, config: TextPreprocessingConfig):
        self.config = config
        self.preprocessor = TextPreprocessor(
            lowercase=config.lowercase, 
            keep_ellie=config.keep_ellie
        )
        
        if config.tokenizer == "whitespace":
            self.tokenizer = WhitespaceTokenizer()
        elif config.tokenizer == "regex":
            self.tokenizer = SimpleRegexTokenizer()
        else:
            raise ValueError(f"Unsupported Tokenizer config requested: {config.tokenizer}")
            
        self.vocabulary = Vocabulary(vocab_size=config.vocab_size)

    def fit(self, text_data_list: List[Dict[str, pd.DataFrame]]) -> None:
        """
        Aggregates global linguistic rules sequentially iterating exclusively over Training bounds natively.
        """
        logger.info("Initializing Text Pipeline Native Fit sequence...")
        tokenized_transcripts = []
        
        for idx, text_data in enumerate(text_data_list):
            try:
                cleaned_text = self.preprocessor.process_participant(idx, text_data)
                tokens = self.tokenizer.transform(cleaned_text)
                tokenized_transcripts.append(tokens)
            except Exception as e:
                logger.error(f"Participant {idx} failed during Text vocab fitting loop: {e}")
                
        if not tokenized_transcripts:
            raise ValueError("Pipeline fit failed natively - no transcripts bypassed filter rules.")
            
        self.vocabulary.build_vocabulary(tokenized_transcripts)
        logger.info(f"Pipeline Text Extraction successful. Vocabulary built.")

    def transform(self, participant_id: int, text_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consumes granular participant dictionaries outputting sequence tensors iteratively.
        """
        cleaned_text = self.preprocessor.process_participant(participant_id, text_data)
        tokens = self.tokenizer.transform(cleaned_text)
        
        logger.info(
            f"Participant {participant_id} | Original Tokens extracted: {len(tokens)}"
        )
        
        token_ids = self.vocabulary.convert_tokens_to_ids(tokens)
        pad_id = self.vocabulary.word2id.get(self.vocabulary.pad_token, 0)
        
        t_ids, t_mask, t_len = prepare_sequence(
            token_ids, 
            max_length=self.config.max_sequence_length, 
            pad_id=pad_id
        )
        
        logger.info(
            f"Participant {participant_id} | Text Arrays Formatted -> "
            f"IDs: {list(t_ids.shape)}, MSK: {list(t_mask.shape)}, LEN: {t_len.item()}"
        )
        
        return {
            "token_ids": t_ids,
            "attention_mask": t_mask,
            "sequence_length": t_len,
            "raw_text": cleaned_text
        }

    def fit_transform(self, participant_id: int, text_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience isolation method."""
        cleaned_text = self.preprocessor.process_participant(participant_id, text_data)
        tokens = self.tokenizer.transform(cleaned_text)
        
        self.vocabulary.build_vocabulary([tokens])
        
        token_ids = self.vocabulary.convert_tokens_to_ids(tokens)
        pad_id = self.vocabulary.word2id.get(self.vocabulary.pad_token, 0)
        
        t_ids, t_mask, t_len = prepare_sequence(
            token_ids, 
            self.config.max_sequence_length, 
            pad_id
        )
        
        return {
            "token_ids": t_ids,
            "attention_mask": t_mask,
            "sequence_length": t_len,
            "raw_text": cleaned_text
        }

    def save_vocabulary(self, path_str: str) -> None:
        self.vocabulary.save(Path(path_str))

    def load_vocabulary(self, path_str: str) -> None:
        self.vocabulary.load(Path(path_str))
