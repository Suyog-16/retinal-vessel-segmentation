import streamlit as st
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import Unet
BASE_DIR = os.path.dirname(__file__)  # path to /app
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_unet.pth")

# Set up
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Unet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Transform
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
    ToTensorV2()
])

# Title
st.title("Retinal Vessel Segmentation")

# Upload or select
st.sidebar.header("Input Image")
uploaded_file = st.sidebar.file_uploader("Upload a retinal image", type=["png", "jpg", "jpeg", "tif"])

test_dir = "C:/Users/Acer nitro/Desktop/retinal-vessel-segmentation/data/DRIVE/test/images"
test_images = [f for f in os.listdir(test_dir) if f.endswith((".tif", ".jpg", ".png"))]
selected_test = st.sidebar.selectbox("...or choose from test images", ["None"] + test_images)

# Load image
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    filename = "Uploaded Image"
elif selected_test != "None":
    image_path = os.path.join(test_dir, selected_test)
    image = Image.open(image_path).convert("RGB")
    filename = selected_test
else:
    st.warning("Upload or select an image to start.")
    st.stop()

# Preprocess
image_np = np.array(image)
transformed = transform(image=image_np)
input_tensor = transformed['image'].unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    pred = torch.sigmoid(model(input_tensor))
    pred_mask = (pred > 0.5).float().squeeze().cpu().numpy()

# Display
st.subheader(f"Original and Predicted Mask: {filename}")
col1, col2 = st.columns(2)
with col1:
    st.image(image, caption="Original Image", use_column_width=True)
with col2:
    st.image(pred_mask, caption="Predicted Vessel Mask", use_column_width=True, clamp=True)
