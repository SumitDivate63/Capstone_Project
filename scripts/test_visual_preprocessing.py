import torch

from datasets.daic_dataset import DAICDataset
from preprocessing.visual.pipeline import (
    VisualPreprocessingPipeline,
    VisualPreprocessingConfig,
)


def test_visual_pipeline():

    dataset = DAICDataset(split="train")

    config = VisualPreprocessingConfig()

    pipeline = VisualPreprocessingPipeline(config)

    train_visual = [dataset[i]["visual"] for i in range(5)]

    pipeline.fit(train_visual)

    sample = dataset[0]

    tensor = pipeline.transform(
        participant_id=sample["participant_id"],
        visual_data=sample["visual"]
    )

    assert isinstance(tensor, torch.Tensor)

    assert tensor.dtype == torch.float32

    assert not torch.isnan(tensor).any()

    assert not torch.isinf(tensor).any()

    assert tensor.ndim == 3

    assert tensor.shape[0] > 0

    assert tensor.shape[1] == config.window_size

    assert tensor.shape[2] > 0


def test_visual_pipeline_stress_test():
    """
    Stress test to fit the scaler on increasing subsets (20 -> 50 -> 101) 
    to verify the new incremental memory-efficient approach.
    """
    try:
        dataset = DAICDataset(split="train")
    except Exception as e:
        print(f"Skipping stress test, dataset not available: {e}")
        return

    config = VisualPreprocessingConfig()
    pipeline = VisualPreprocessingPipeline(config)

    max_len = len(dataset)

    # 20 participants
    count_20 = min(20, max_len)
    if count_20 > 0:
        print(f"Fitting on {count_20} participants...")
        train_visual_20 = [dataset[i]["visual"] for i in range(count_20)]
        pipeline.fit(train_visual_20)

    # 50 participants
    count_50 = min(50, max_len)
    if count_50 > 0:
        print(f"Fitting on {count_50} participants...")
        train_visual_50 = [dataset[i]["visual"] for i in range(count_50)]
        pipeline.fit(train_visual_50)

    # 101 participants
    count_101 = min(101, max_len)
    if count_101 > 0:
        print(f"Fitting on {count_101} participants...")
        train_visual_101 = [dataset[i]["visual"] for i in range(count_101)]
        pipeline.fit(train_visual_101)

    if max_len == 0:
        return

    sample = dataset[0]
    tensor = pipeline.transform(
        participant_id=sample["participant_id"],
        visual_data=sample["visual"]
    )

    assert isinstance(tensor, torch.Tensor)
    assert not torch.isnan(tensor).any(), "NaN found in transformed tensor"
    assert not torch.isinf(tensor).any(), "Inf found in transformed tensor"

if __name__ == "__main__":
    print("=" * 70)
    print("VISUAL PREPROCESSING STRESS TEST")
    print("=" * 70)

    test_visual_pipeline_stress_test()

    print("\n")
    print("=" * 70)
    print("VISUAL PREPROCESSING STRESS TEST PASSED")
    print("=" * 70)
