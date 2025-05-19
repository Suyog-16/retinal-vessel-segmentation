import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import Unet  # Make sure this matches your model definition

def parse_args():
    parser = argparse.ArgumentParser(description="UNet Inference Script")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights (.pth)')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the test image')
    parser.add_argument('--img_size', type=int, default=256, help='Resize image to this size (square)')
    return parser.parse_args()

def preprocess_image(image_path, img_size):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # Converts to [0,1] and channels-first (C,H,W)
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),  # Normalize to [-1, 1]
    ])
    return transform(image).unsqueeze(0), image  # return tensor [1,3,H,W] and original PIL image

def postprocess_output(output_tensor):
    # Output is [1, 1, H, W] — squeeze to [H, W] and apply sigmoid
    output = torch.sigmoid(output_tensor.squeeze()).detach().cpu().numpy()
    return (output > 0.5).astype(np.uint8)  # Binary mask

def main():
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = Unet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Preprocess image
    input_tensor, original_image = preprocess_image(args.image_path, args.img_size)
    input_tensor = input_tensor.to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Post-process
    mask = postprocess_output(output)

    # Display results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("Predicted Mask")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

