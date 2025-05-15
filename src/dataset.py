import os
import PIL as Image
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

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        mask_np = np.array(mask)
        mask_bin = (mask_np > 0).astype(np.uint8)

        #if index == 0:
            #print("Unique mask values (raw):", np.unique(mask_np))
            #print("Unique mask values (binarized):", np.unique(mask_bin))

        mask = Image.fromarray(mask_bin)

        if self.transform:
            seed = random.randint(0, 1000000)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)
            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.transform(mask)
            mask = (mask >= 0.003921568).float()

        #if index == 0:
            #print("Mask sum after transform:", mask.sum().item())
            #print("Unique mask values after transform:", torch.unique(mask).tolist())

        return image, mask