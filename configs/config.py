from typing import List
import warnings

from pydantic import BaseModel, validator
from omegaconf import OmegaConf

class LossConfig(BaseModel):
    alias: str
    weight: float
    loss_fn: str
    loss_kwargs: dict


    @validator('weight')
    def check_weight(cls, value):
        if value < 0:
            raise ValueError('Weight for loss must be between 0 and 1 inclusive')
        return value

class TrainSettings(BaseModel):
    batch_size: int
    n_workers: int
    train_test_split: float
    img_h: int
    img_w: int
    random_seed: int

    @validator('batch_size')
    def check_batch_size(cls, value):
        if value <= 1:
            warnings.warn("Batch is <= 1. May result in slow speed", UserWarning)
        return value

class Model(BaseModel):
    name: str
    encoder_name: str
    encoder_weights: str
    in_channels: int
    num_cls: int

class Experiments(BaseModel):
    project_name: str
    experiment_name: str
    training_settings: TrainSettings
    n_epochs: int
    lr: float
    accelerator: str
    device: int
    optimizer_param: dict
    scheduler_param: dict
    cls_losses: List[LossConfig]
    seg_losses: List[LossConfig]

    @classmethod
    def load_yaml(cls, path: str) -> "Experiments":
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)