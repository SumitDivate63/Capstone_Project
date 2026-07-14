from datasets.daic_dataset import DAICDataset
from preprocessing.text.pipeline import (
    TextPreprocessingPipeline,
    TextPreprocessingConfig,
)

import torch

print("=" * 70)
print("TEXT PREPROCESSING PIPELINE TEST")
print("=" * 70)

# -----------------------------
# Load Dataset
# -----------------------------
dataset = DAICDataset(split="train")

print(f"Dataset Size : {len(dataset)}")

sample = dataset[0]

print(f"Participant : {sample['participant_id']}")

print("\nTranscript Columns:")
print(sample["text"]["transcript"].columns)

# -----------------------------
# Create Pipeline
# -----------------------------
config = TextPreprocessingConfig(
    max_sequence_length=512
)

pipeline = TextPreprocessingPipeline(config)

# -----------------------------
# Fit Vocabulary
# -----------------------------
print("\nBuilding Vocabulary...")

train_text = [
    dataset[i]["text"]
    for i in range(min(20, len(dataset)))
]

pipeline.fit(train_text)

print("Vocabulary Built.")

# -----------------------------
# Transform One Participant
# -----------------------------
output = pipeline.transform(
    sample["participant_id"],
    sample["text"]
)

print()
print("=" * 70)
print("TEXT")
print("=" * 70)

print("Keys :", output.keys())
print()

print("Token IDs Shape :", output["token_ids"].shape)
print("Attention Mask Shape :", output["attention_mask"].shape)
print("Sequence Length :", output["sequence_length"])
print()

print("Token Tensor Type :", type(output["token_ids"]))
print("Attention Type :", type(output["attention_mask"]))
print()

print("NaN Count :", torch.isnan(output["token_ids"].float()).sum())
print("Inf Count :", torch.isinf(output["token_ids"].float()).sum())
print()

print("Vocabulary Size :", len(pipeline.vocabulary.word2id))
print()

print("Raw Text Preview:")
print(output["raw_text"][:300])
print()

print("=" * 70)
print("TEXT PREPROCESSING PASSED")
print("=" * 70)
