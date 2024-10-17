import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2


def get_train_augmentations(img_width: int, img_height: int) -> alb.Compose:
    return alb.Compose([
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        alb.ShiftScaleRotate(),
        alb.GaussianBlur(),
        alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        alb.Resize(height=img_height, width=img_width),
        alb.Normalize(),
        ToTensorV2()
    ])

def get_validation_augmentations(img_width: int, img_height: int) -> alb.Compose:
    return alb.Compose([alb.Resize(height=img_height, width=img_width), alb.Normalize(), ToTensorV2()])