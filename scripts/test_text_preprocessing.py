import pandas as pd
import torch
import tempfile
import logging
from pathlib import Path

from preprocessing.text.pipeline import TextPreprocessingPipeline, TextPreprocessingConfig

def test_pipeline():
    config = TextPreprocessingConfig(max_sequence_length=16, vocab_size=100)
    pipeline = TextPreprocessingPipeline(config)
    
    data = {
        "speaker": ["Ellie", "Participant", "Ellie", "Participant"],
        "value": ["Hi", "Hello there.", "How are you.", "I'm doing well!"]
    }
    df = pd.DataFrame(data)
    participant_data = {"transcript": df}
    
    # Check Pipeline fit
    pipeline.fit([participant_data])
    assert pipeline.vocabulary.is_fitted
    
    # Check Pipeline Transform
    output = pipeline.transform(999, participant_data)
    
    assert "token_ids" in output
    assert "attention_mask" in output
    assert "sequence_length" in output
    assert "raw_text" in output
    
    ids_tensor = output["token_ids"]
    mask = output["attention_mask"]
    
    assert isinstance(ids_tensor, torch.LongTensor)
    assert isinstance(mask, torch.LongTensor)
    
    assert ids_tensor.shape == (16,)
    assert mask.shape == (16,)
    
    assert not torch.isnan(ids_tensor.float()).any()
    assert output["sequence_length"].item() > 0
    
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = Path(tmpdir) / "vocab.pkl"
        pipeline.save_vocabulary(str(vocab_path))
        
        new_pipeline = TextPreprocessingPipeline(config)
        new_pipeline.load_vocabulary(str(vocab_path))
        
        assert new_pipeline.vocabulary.is_fitted
        assert new_pipeline.vocabulary.word2id == pipeline.vocabulary.word2id

    print("All Text Pipeline unit tests passed natively.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_pipeline()
