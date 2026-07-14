from datasets.daic_dataset import DAICDataset
from preprocessing.audio.pipeline import (
    AudioPreprocessingPipeline,
    AudioPreprocessingConfig,
)
import torch

dataset = DAICDataset(split="train")

config = AudioPreprocessingConfig()
pipeline = AudioPreprocessingPipeline(config)

train_audio = [dataset[i]["audio"] for i in range(5)]
pipeline.fit(train_audio)

sample = dataset[0]

tensor = pipeline.transform(
    participant_id=sample["participant_id"],
    audio_data=sample["audio"]
)

print("=" * 60)
print("AUDIO PREPROCESSING VALIDATION")
print("=" * 60)

print("Tensor Shape :", tensor.shape)
print("Tensor Type  :", type(tensor))
print("Tensor Dtype :", tensor.dtype)

print()

print("NaN Count    :", torch.isnan(tensor).sum())
print("Inf Count    :", torch.isinf(tensor).sum())

print()

print("Mean         :", tensor.mean())
print("Std          :", tensor.std())

print()

print("Minimum      :", tensor.min())
print("Maximum      :", tensor.max())

pipeline.save_scaler("outputs/checkpoints/audio_scaler.pkl")
print("Scaler Saved Successfully")

pipeline.load_scaler("outputs/checkpoints/audio_scaler.pkl")
print("Scaler Loaded Successfully")
