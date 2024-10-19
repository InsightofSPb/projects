
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.dataset import BarcodeDataset
from configs.config import TrainSettings
from src.augmentations import get_train_augmentations, get_validation_augmentations
from src.utils import read_dataset_data

from src.paths import IMAGES_PATH

class BarcodeDataModule(LightningDataModule):
    def __init__(self, cfg: TrainSettings) -> None:
        super().__init__()

        self.batch_size = cfg.batch_size
        self.n_workers = cfg.n_workers
        self.train_test_split = cfg.train_test_split
        self.train_augmentations = get_train_augmentations(cfg.img_w, cfg.img_h)
        self.val_test_augmentations = get_validation_augmentations(cfg.img_w, cfg.img_h)
        self.random_seed = cfg.random_seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None) -> None:
        df = read_dataset_data()
        train_df, test_df = train_test_split(df, train_size=self.train_test_split, random_state=self.random_seed)
        val_df, test_df = train_test_split(test_df, train_size=0.5, random_state=self.random_seed)

        if stage == 'fit' or stage is None:
            self.train_dataset = BarcodeDataset(train_df, IMAGES_PATH, self.train_augmentations)
            self.val_dataset = BarcodeDataset(val_df, IMAGES_PATH, self.val_test_augmentations)

        if stage == 'test' or stage is None:
            self.test_dataset = BarcodeDataset(test_df, IMAGES_PATH, self.val_test_augmentations)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )


