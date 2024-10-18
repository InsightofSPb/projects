from typing import List

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


