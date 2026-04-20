import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config


def get_train_transforms():
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class ISICDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        mask = mask.unsqueeze(0) if isinstance(mask, torch.Tensor) else torch.tensor(mask).unsqueeze(0)
        return img, mask.float()


def get_loaders():
    images = sorted([
        os.path.join(config.IMAGE_DIR, f)
        for f in os.listdir(config.IMAGE_DIR)
        if f.endswith((".jpg", ".png"))
    ])

    masks = []
    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        mask_name = name + "_segmentation.png"
        mask_path = os.path.join(config.MASK_DIR, mask_name)
        masks.append(mask_path)

    valid_pairs = [(i, m) for i, m in zip(images, masks) if os.path.exists(m)]
    images, masks = zip(*valid_pairs)
    images, masks = list(images), list(masks)

    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
        images, masks, test_size=(1 - config.TRAIN_RATIO), random_state=config.SEED
    )
    relative_val = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        temp_imgs, temp_masks, test_size=(1 - relative_val), random_state=config.SEED
    )

    print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

    train_ds = ISICDataset(train_imgs, train_masks, transform=get_train_transforms())
    val_ds = ISICDataset(val_imgs, val_masks, transform=get_val_transforms())
    test_ds = ISICDataset(test_imgs, test_masks, transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader, test_ds
