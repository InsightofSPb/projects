import cv2
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from augmentations import get_train_augmentations
from paths import IMAGES_PATH, ANNOTATION_PATH



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
    annotations = pd.read_csv(os.path.join(ANNOTATION_PATH, 'annotations.csv'), sep='\t')
    folder_dir = IMAGES_PATH

    img_width, img_height = 224, 224
    train_transforms = get_train_augmentations(img_width, img_height)

    dataset = BarcodeDataset(annotations_file=annotations, folder_dir=folder_dir, transforms=train_transforms)

    img_width, img_height = 224, 224

    num_images = 5

    output_dir = './augmented_images'
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_images):
        idx = np.random.randint(0, len(dataset))
        img, mask = dataset[idx]
        img_np = img.permute(1, 2, 0).numpy()
        img_np = img_np * 255 
        img_np = img_np.astype(np.uint8)

        mask_np = mask.numpy()
        mask_np = (mask_np * 255).astype(np.uint8)
        overlay = cv2.addWeighted(img_np, 0.8, cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR), 0.2, 0)

        output_path = os.path.join(output_dir, f"augmented_image_{i+1}.png")
        plt.imsave(output_path, overlay)
        print(f"Сохранено: {output_path}")


if __name__ == "__main__":
    main()