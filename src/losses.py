from dataclasses import dataclass
from typing import List

from torch import nn
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

from configs.config import LossConfig
from utils import load_loss

@dataclass
class Loss:
    alias: str
    weight: float
    entity: nn.Module


def use_loss(cfgs: List[LossConfig]) -> List[Loss]:
    return [
        Loss(
            alias=cfg.alias,
            weight=cfg.weight,
            loss=load_loss(cfg.loss)
        ) for cfg in cfgs
    ]


def main():
    bce = Loss('bce', 0.3, nn.BCELoss())
    dice = Loss('dice', 0.4, DiceLoss('binary'))
    focal = Loss('focal', 0.3, FocalLoss('binary'))
    return [bce, dice, focal]

if __name__ == "__main__":
    print(main())
