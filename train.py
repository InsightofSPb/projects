import os
import torch
import pytorch_lightning

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from clearml import Task

from configs.config import Experiments, Model
from src.paths import CONFIG_PATH, EXPERIMENTS_PATH
from src.datamodule import BarcodeDataModule
from src.main_module import BarcodeModule



def train(cfg: Experiments, cfg_model: Model):
    if cfg_model.clearml:
        task = Task.init(
            project_name=cfg.project_name,
            task_name=cfg.experiment_name,
            output_uri=None
        )
        
        task.connect(cfg.dict())
    
    else:
        task = None

    datamodule = BarcodeDataModule(cfg=cfg.training_settings)
    model = BarcodeModule(cfg_model, cfg_exp=cfg)
    res_path = os.path.join(EXPERIMENTS_PATH, cfg.experiment_name)
    os.makedirs(res_path, exist_ok=True)
    metric = cfg.monitor_metric

    chckp_callback = ModelCheckpoint(
        dirpath=res_path,
        monitor=metric,
        mode=cfg.monitor_mode,
        save_top_k=1,
        filename=f'{cfg_model.name}_encoder-{cfg_model.encoder_name}_epoch_{{epoch:02d}}-{metric}_{{{metric}:.3f}}'
    )

    loggers = []

    if cfg_model.wandb:
        wandb_logger = WandbLogger(
            project=cfg.project_name,
            name=cfg.experiment_name,
            log_model=False,
            config = cfg.dict()
        )
        loggers.append(wandb_logger)

    trainer = pytorch_lightning.Trainer(
        max_epochs=cfg.n_epochs,
        accelerator=cfg.accelerator,
        devices=[cfg.device],
        logger=loggers if loggers else None,
        callbacks=[
            chckp_callback,
            EarlyStopping(monitor=cfg.monitor_metric, patience=4, mode=cfg.monitor_mode),
            LearningRateMonitor(logging_interval='epoch')
        ],
        deterministic=False,
        log_every_n_steps=20
    )

    trainer.fit(model, datamodule)
    trainer.test(ckpt_path=chckp_callback.best_model_path, datamodule=datamodule)

    if task:
        task.upload_artifact('best_model', chckp_callback.best_model_path)
        task.close()


if __name__ == "__main__":
    cfg = Experiments.load_yaml(os.path.join(CONFIG_PATH, 'train_cfg.yaml'))
    cfg_m = Model.load_yaml(os.path.join(CONFIG_PATH, 'model_cfg.yaml'))
    seed_everything(cfg.training_settings.random_seed, workers=True)
    train(cfg, cfg_m)