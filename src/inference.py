import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from model import Unet
from torch import utils

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
    ToTensorV2()
])

## loading the saved model
model = Unet(in_channels=3,out_channels= 1)

model.load_state_dict(torch.load("C:/Users/Acer nitro/Desktop/retinal-vessel-segmentation/models/best_unet.pth", map_location=device))
model.to(device)
model.eval()

#loading and preprocessing the image

img_path = "C:/Users/Acer nitro/Desktop/retinal-vessel-segmentation/data/DRIVE/test/images/01_test.tif"

image = Image.open(img_path).convert("RGB")
image_np = np.array(image)


transformed = transform(image = image_np)
input_tensor = transformed['image'].unsqueeze(dim = 0).to(device)

print("Model loaded and ready.")
print("Starting inference...")
# inference
with torch.no_grad():
    pred = torch.sigmoid(model(input_tensor))
    pred_mask = (pred > 0.5).float().squeeze().cpu().numpy()


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)

plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(pred_mask,cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")

plt.tight_layout()
plt.show()
