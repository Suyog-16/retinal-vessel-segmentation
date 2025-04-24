import os
import PIL as Image
from torch.utils.data import Dataset

class DRIVE_dataset(Dataset):
    def __init__(self,img_dir,mask_dir,transforms = None, mask_transfrom = None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.mask_transform = mask_transfrom
        self.images = os.listdir(img_dir)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.img_dir,image_name)
        mask_path = os.path.join(self.mask_dir,image_name) # masks also have the same name as the image

        image = Image.open(image_path).convert("RGB")        
        mask = Image.open(mask_path)

        if self.transforms:
            image = self.transforms(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image,mask