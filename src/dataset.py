import cv2
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset



class BarcodeDataset(Dataset):
    def __init__(self, annotations_file, folder_dir, transforms=None) -> None:
        self.labels = annotations_file
        self.dir = folder_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = os.path.basename(self.labels.iloc[idx, 0])
        img_path = os.path.join(self.dir, img_name)
        img = cv2.imread(str(img_path))
        
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x, y, w, h = self.labels.iloc[idx, 2:6]
        mask = np.zeros(img.shape[:2], dtype=np.float32)
        mask[y:y + h, x:x + w] = 1.0

        if self.transforms:
            augments = self.transforms(image=img, mask=mask)
            img = augments["image"]
            mask = augments["mask"]
        
        return img, mask


def main():
    annotations = pd.read_csv("/home/sasha/segment_barcode/data/annotations.csv", sep='\t')
    folder_dir = '/home/sasha/segment_barcode/data/images'

    dataset = BarcodeDataset(annotations_file=annotations, folder_dir=folder_dir)

    for i in range(min(len(dataset), 3)): 
        img, mask = dataset[i]
        print(f"Image {i}: shape {img.shape}, Mask {i}: shape {mask.shape}")


if __name__ == "__main__":
    main()