import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from model import Unet
from torch import utils
import argparse
#----------------Define and Parse CLI arguments--------------------
parser = argparse.ArgumentParser(description = "Inference with trained model")
parser.add_argument("--model_path",type=str,required=True, help='Path to model')
parser.add_argument("--image_path",type = str, required = True, help = 'Path to the test image')

args = parser.parse_args()

#---------------- Set Device -------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

#---------------- Define Transforms-------------------------------
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
    ToTensorV2()
])

#----------------Load Models-----------------------------------------
model = Unet(in_channels=3,out_channels= 1)

model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()

#---------------Loading and preprocessing the image-------------------------

img_path = args.image_path

image = Image.open(img_path).convert("RGB")
image_np = np.array(image)


transformed = transform(image = image_np)
input_tensor = transformed['image'].unsqueeze(dim = 0).to(device)

print("Model loaded and ready.")
print("Starting inference...")
#--------------------Inference--------------------------------
with torch.no_grad():
    pred = torch.sigmoid(model(input_tensor))
    pred_mask = (pred > 0.5).float().squeeze().cpu().numpy()

#--------------------Plot the results-------------------------
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
