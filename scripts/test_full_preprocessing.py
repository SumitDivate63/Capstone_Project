from datasets.daic_dataset import DAICDataset

from preprocessing.visual.pipeline import (
    VisualPreprocessingPipeline,
    VisualPreprocessingConfig,
)

from preprocessing.audio.pipeline import (
    AudioPreprocessingPipeline,
    AudioPreprocessingConfig,
)

import torch

print("=" * 70)
print("FULL PREPROCESSING PIPELINE TEST")
print("=" * 70)

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------
dataset = DAICDataset(split="train")

print(f"Dataset Size : {len(dataset)}")

sample = dataset[0]

print(f"Participant : {sample['participant_id']}")

# ----------------------------------------------------
# Initialize Pipelines
# ----------------------------------------------------
visual_pipeline = VisualPreprocessingPipeline(
    VisualPreprocessingConfig()
)

audio_pipeline = AudioPreprocessingPipeline(
    AudioPreprocessingConfig()
)

# ----------------------------------------------------
# Fit Pipelines
# ----------------------------------------------------
print("\nFitting Visual Pipeline...")

visual_train = [dataset[i]["visual"] for i in range(5)]
visual_pipeline.fit(visual_train)

print("Visual Pipeline Ready.")

print("\nFitting Audio Pipeline...")

audio_train = [dataset[i]["audio"] for i in range(5)]
audio_pipeline.fit(audio_train)

print("Audio Pipeline Ready.")

# ----------------------------------------------------
# Transform Sample
# ----------------------------------------------------
print("\nTransforming Participant...")

visual_tensor = visual_pipeline.transform(
    participant_id=sample["participant_id"],
    visual_data=sample["visual"],
)

audio_tensor = audio_pipeline.transform(
    participant_id=sample["participant_id"],
    audio_data=sample["audio"],
)

# ----------------------------------------------------
# Results
# ----------------------------------------------------
print("\n" + "=" * 70)
print("VISUAL")
print("=" * 70)

print("Shape :", visual_tensor.shape)
print("Type  :", type(visual_tensor))
print("NaN   :", torch.isnan(visual_tensor).sum().item())
print("Inf   :", torch.isinf(visual_tensor).sum().item())

print()

print("=" * 70)
print("AUDIO")
print("=" * 70)

print("Shape :", audio_tensor.shape)
print("Type  :", type(audio_tensor))
print("NaN   :", torch.isnan(audio_tensor).sum().item())
print("Inf   :", torch.isinf(audio_tensor).sum().item())

print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)

assert torch.isnan(visual_tensor).sum() == 0
assert torch.isinf(visual_tensor).sum() == 0

assert torch.isnan(audio_tensor).sum() == 0
assert torch.isinf(audio_tensor).sum() == 0

print("✓ Visual preprocessing passed.")
print("✓ Audio preprocessing passed.")
print("✓ Full preprocessing pipeline passed.")
print("=" * 70)
