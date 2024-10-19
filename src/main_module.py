from typing import Any
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
    

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.exp_cfg.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.8,
            patience=2,
            min_lr=1e-6
            )
        
        return {
            'optimizer' : optimizer,
            'lr_scheduler' : {
                'scheduler' : scheduler,
                'monitor' : 'val_loss',
                'interval' : 'epoch',
                'frequency' : 1,
            }
        }
    
    def calc_loss(self, pred_masks_logits: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        for elem in self.seg_losses:
            loss = elem.loss(pred_masks_logits, gt_masks)
            total_loss += elem.weight * loss
        
        return total_loss
    
    def training_step(self, batch):
        images, gt_masks = batch
        pred_masks_logits = self(images)
        loss = self.calc_loss(pred_masks_logits, gt_masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch):
        images, gt_masks = batch
        gt_masks = gt_masks.long().unsqueeze(1)
        pred_masks_logits = self(images)

        loss = self.calc_loss(pred_masks_logits, gt_masks)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        pred_masks = torch.sigmoid(pred_masks_logits)
        self.val_seg_metrics.update(pred_masks, gt_masks)

    def test_step(self, batch):
        images, gt_masks = batch
        gt_masks = gt_masks.long().unsqueeze(1)
        pred_masks_logits = self(images)
        pred_masks = torch.sigmoid(pred_masks_logits)
        self._test_seg_metrics.update(pred_masks, gt_masks)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_seg_metrics.compute()
        for k, v in metrics.items():
            self.log(f'val_{k}', v, on_epoch=True)
        self.val_seg_metrics.reset()
    
    def on_validation_epoch_end(self) -> None:
        metrics = self.test_seg_metrics.compute()
        for k, v in metrics.items():
            self.log(f'test_{k}', v, on_epoch=True)
        self.test_seg_metrics.reset()