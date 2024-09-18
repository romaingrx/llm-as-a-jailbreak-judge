import random
from typing import List

from dspy import Example
from dspy.datasets import DataLoader
from omegaconf import DictConfig

dl = DataLoader()


def load_dataset(cfg: DictConfig) -> List[Example]:
    dataset = dl.from_csv(
        cfg.dataset.file,
        fields=["goal", "prompt", "response", "jailbroken"],
        input_keys=("goal", "prompt", "response"),
    )
    if cfg.dataset.limit:
        random.seed(cfg.seed)
        sample = random.sample(dataset, cfg.dataset.limit)
        return sample
    return dataset


