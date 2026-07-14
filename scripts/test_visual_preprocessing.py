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
