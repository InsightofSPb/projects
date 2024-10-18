from typing import List
import warnings

from pydantic import BaseModel, validator


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
