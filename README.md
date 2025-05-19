# 🩺 Retinal Vessel Segmentation using U-Net

This project performs **semantic segmentation** of blood vessels in retinal images using a U-Net architecture, trained on the [DRIVE](https://drive.grand-challenge.org/) dataset. The goal is to assist in diagnosing diabetic retinopathy and related conditions.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Suyog-16/retinal-vessel-segmentation.git
cd retinal-vessel-segmentation
```

### 2. Install Dependencies

Use a virtual environment:

```bash
pip install -r requirements.txt
```

---

## 🧠 Training

Place the DRIVE dataset inside the `dataset/` folder in the following format:

```
dataset/
├── training/
│   ├── images/
│   └── masks/
├── test/
│   ├── images/
│   └── masks/
```

Then run:

```bash
python src/train.py
```

This will train the U-Net model and save the best model checkpoint in `models/unet_best.pth`.

---

## 🔍 Inference

You can test your trained model on a new image using the inference script.

### 1. Place test images

Put the image(s) you want to test in a folder, for example:

```
samples/
└── example_image.png
```

### 2. Run inference

```bash
python src/inference.py --img_path samples/example_image.png --model_path models/unet_weights.pth
```

### 3. Output

- The segmentation result will be displayed using matplotlib.
- You can modify the script to save the output to disk if needed.

---

## ⚙️ Inference Script Arguments

| Argument      | Description                            | Default              |
|---------------|----------------------------------------|----------------------|
| `--img_path`  | Path to the input image                | Required             |
| `--model_path`| Path to the trained model `.pth` file  | `models/unet_weights.pth` |

---

## 📈 Evaluation Metrics

- **Dice Score**: ~0.71 on DRIVE test set
- **IoU Score**: ~0.55 on DRIVE test set

---

## 📌 Features

- U-Net architecture in PyTorch
- Combined BCE + Dice Loss for better performance
- Data augmentation with Albumentations
- Easily extendable to web demo with Streamlit

---

## 📷 Sample Results

![Sample Input](samples/example_image.png)
![Prediction](samples/example_output.png)

---

## 🤝 Acknowledgments

- Dataset: [DRIVE](https://drive.grand-challenge.org/)
- U-Net: Ronneberger et al., 2015
- Augmentations: [Albumentations](https://albumentations.ai)

---

## 📜 License

MIT License# 

