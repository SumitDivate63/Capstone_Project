import re
import pandas as pd
import unicodedata
from utils.logger import get_logger

logger = get_logger(__name__)

def clean_text(text: str, lowercase: bool = True) -> str:
    """
    Cleans transcript text by standardizing spaces, punctuation and case.
    """
    if not isinstance(text, str):
        return ""
    if lowercase:
        text = text.lower()
        
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\s+', ' ', text)
    # Remove repeated punctuation
    text = re.sub(r'([.?!,])\1+', r'\1', text)
    return text.strip()

def remove_non_dialogue(df: pd.DataFrame, keep_ellie: bool = False) -> pd.DataFrame:
    """
    Removes timestamps, AI agent dialogue (if configured), and nulls.
    """
    df = df.copy()
    if 'speaker' in df.columns:
        if not keep_ellie:
            df = df[df['speaker'].astype(str).str.lower() != 'ellie']
    if 'value' in df.columns:
        df = df.dropna(subset=['value'])
    return df

def merge_utterances(df: pd.DataFrame, keep_ellie: bool = False) -> str:
    """
    Merges filtered transcript values into a single text document.
    """
    df = remove_non_dialogue(df, keep_ellie=keep_ellie)
    if 'value' not in df.columns:
        return ""
    return " ".join([str(val).strip() for val in df['value'].tolist() if str(val).strip()])

def validate_transcript(text: str) -> None:
    """
    Ensures text is not empty and has minimum logical length.
    """
    if not text or not text.strip():
        raise ValueError("Transcript is completely empty after cleaning.")
    tokens = text.split()
    if len(tokens) < 1:
        raise ValueError(f"Transcript contains too few tokens: {len(tokens)}")
