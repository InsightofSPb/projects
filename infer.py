import os
import torch
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from configs.config import InferenceConfig, ModelInference
from src.paths import CONFIG_PATH
from src.datamodule import BarcodeDataModule
from src.main_module import BarcodeModule

def load_model(cfg: InferenceConfig, model_cfg: ModelInference, checkpoint_path: str):
    model = BarcodeModule.load_from_checkpoint(checkpoint_path, cfg_model=model_cfg, cfg_exp=cfg, inference=True)
    return model

def save_inference_results(original_img, predicted_mask, save_dir, image_name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_img.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original Image")
    
    axes[1].imshow(predicted_mask.cpu().numpy(), cmap='gray')
    axes[1].set_title("Predicted Mask")
    
    plt.savefig(os.path.join(save_dir, f"{image_name}_result.png"))
    plt.close()

def run_inference(cfg: InferenceConfig, model_cfg: ModelInference):
    datamodule = BarcodeDataModule(cfg=cfg.inference_settings, inference=True)
    
    model = load_model(cfg, model_cfg, cfg.inference_settings.checkpoint_path)
    
    trainer = Trainer(accelerator=cfg.device_settings.accelerator, devices=[cfg.device_settings.device])
    
    predictions = trainer.predict(model=model, datamodule=datamodule.test_dataloader)
    
    output_dir = cfg.inference_settings.output_path
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, (batch_images, batch_preds) in enumerate(zip(datamodule.dataset, predictions)):
        for i in range(len(batch_images)):
            original_img = batch_images[i]
            predicted_mask = torch.sigmoid(batch_preds[i]) > 0.5 
            
            save_inference_results(original_img, predicted_mask, output_dir, f"image_{idx}_{i}")

if __name__ == "__main__":
    cfg = InferenceConfig.load_yaml(os.path.join(CONFIG_PATH, 'inference.yaml'))
    model_cfg = ModelInference.load_yaml(os.path.join(CONFIG_PATH, 'model_cfg.yaml'))
    
    run_inference(cfg, model_cfg)
