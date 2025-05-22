import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import numpy as np

class DRIVE_dataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [img for img in os.listdir(img_dir) if img.endswith('.tif')]
        self.valid_pairs = []
        for img in self.images:
            if 'training' in img or 'test' in img:
                base_name = img.split('_')[0]
                mask_name = f"{base_name}_manual1.gif"
                mask_path = os.path.join(self.mask_dir, mask_name)
                if os.path.exists(mask_path):
                    self.valid_pairs.append((img, mask_name))
        if not self.valid_pairs:
            raise ValueError("No valid image-mask pairs")
        print(f"Found {len(self.valid_pairs)} valid pairs in {img_dir}")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, index):
        image_name, mask_name = self.valid_pairs[index]
        image_path = os.path.join(self.img_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Binarize mask: make sure it's 0 and 1
        mask = (mask > 0).astype(np.uint8)

        # If needed, expand mask dims to HWC
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.float()
            mask = mask.permute(2,0,1)

        return image, mask
