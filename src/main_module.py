from pytorch_lightning import LightningModule
import torch

from local_model_zoo import UnetModel, DeepLabV3PlusModel
from configs.config import Model, Experiments
from losses import use_loss
from metrics import get_metrics

class BarcodeModule(LightningModule):
    def __init__(self, cfg_model, cfg_exp) -> None:
        super().__init__()
        self.model_cfg = cfg_model
        self.exp_cfg = cfg_exp
        
        self.model = self._init_model()

        self.seg_losses = use_loss(self.exp_cfg.seg_losses)
        self.val_seg_metrics = get_metrics()
        self.test_seg_metrics = get_metrics()
        self.save_hyperparameters(self.exp_cfg.dict())

    def _init_model(self):
        model_name = self.model_cfg['name']
        encoder_name = self.model_cfg['encoder_name']
        encoder_weights = self.model_cfg['encoder_weights']
        in_channels = self.model_cfg['in_channels']
        num_cls = self.model_cfg['num_cls']

        if model_name == 'Unet':
            return UnetModel(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                num_cls=num_cls
            ).get_model()
        
        elif model_name == 'DeepLabV3+':
            return DeepLabV3PlusModel(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                num_cls=num_cls
            ).get_model()
        
        else:
            raise ValueError(f'Model {model_name} is not presented in model local zoo')
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.model(x)