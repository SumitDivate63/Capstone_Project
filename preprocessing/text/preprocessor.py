import pandas as pd
from typing import Dict, Any
from .cleaning import clean_text, merge_utterances, validate_transcript
from utils.logger import get_logger

logger = get_logger(__name__)


class TextPreprocessor:
    """
    Logic Controller abstracting cleaning complexities and executing filters sequentially.
    """
    def __init__(self, lowercase: bool = True, keep_ellie: bool = False):
        self.lowercase = lowercase
        self.keep_ellie = keep_ellie

    def process_participant(self, participant_id: int, text_data: Dict[str, pd.DataFrame]) -> str:
        """
        Takes raw pandas Transcript dictionaries, strips out non-required features iteratively, and yields pure text.
        """
        if "transcript" not in text_data:
            raise KeyError(f"Participant {participant_id}: Missing raw 'transcript' pandas block in payload.")
            
        df = text_data["transcript"]
        
        merged_text = merge_utterances(df, keep_ellie=self.keep_ellie)
        cleaned_text = clean_text(merged_text, lowercase=self.lowercase)
        validate_transcript(cleaned_text)
        
        logger.info(
            f"Participant {participant_id} | Text cleansing sequence complete. "
            f"Final Raw Extracted char length: {len(cleaned_text)}"
        )
        return cleaned_text
