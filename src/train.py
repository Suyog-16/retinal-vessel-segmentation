import torch
import tqdm
from torch import nn
from dataset import DRIVE_dataset
from utils import dice_loss,dice_score,combined_loss
from torchvision import transforms as T
from torch.utils import DataLoader
from model import Unet
from PIL import Image



device = 'cuda' if torch.cuda.is_available() else "cpu"

def train_step(model, dataloader, criterion, optimizer):
    model.train()
    train_loss, train_dice = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_dice += dice_score(y_pred, y)
    train_loss /= len(dataloader)
    train_dice /= len(dataloader)
    return train_loss, train_dice



def test_step(model, dataloader, criterion, epoch):
    model.eval()
    test_loss, test_dice = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            test_loss += loss.item()
            test_dice += dice_score(y_pred, y)
            if batch == 0:
                pred = torch.sigmoid(y_pred)
                pred = (pred > 0.5).float()
                #utils.save_image(pred, f"pred_epoch_{epoch+1}.png")
                #utils.save_image(y, f"target_epoch_{epoch+1}.png")
    test_loss /= len(dataloader)
    test_dice /= len(dataloader)
    return test_loss, test_dice


def train(model, train_dataloader, test_dataloader, optimizer, criterion, epochs=100):
    results = {"train_loss": [], "train_dice": [], "test_loss": [], "test_dice": []}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    for epoch in tqdm(range(epochs)):
        train_loss, train_dice = train_step(model, train_dataloader, criterion, optimizer)
        test_loss, test_dice = test_step(model, test_dataloader, criterion, epoch)
        print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_dice: {train_dice:.4f} | "
              f"test_loss: {test_loss:.4f} | test_dice: {test_dice:.4f}")
        scheduler.step(test_loss)
        results["train_loss"].append(train_loss)
        results["train_dice"].append(train_dice)
        results["test_loss"].append(test_loss)
        results["test_dice"].append(test_dice)
    return results

if __name__ == "__main__":
    transform = T.Compose([
        T.Resize((256, 256), interpolation=Image.NEAREST),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor()
    ])
    image_transform = T.Compose([
        transform,
        #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_data = DRIVE_dataset(
        img_dir="/content/drive/MyDrive/retina-vessel-segmentation/data/DRIVE/training/images",
        mask_dir="/content/drive/MyDrive/retina-vessel-segmentation/data/DRIVE/training/1st_manual",
        transform=transform
    )
    test_data = DRIVE_dataset(
        img_dir="/content/drive/MyDrive/retina-vessel-segmentation/data/DRIVE/test/images",
        mask_dir="/content/drive/MyDrive/retina-vessel-segmentation/data/DRIVE/test/1st_manual",
        transform=transform
    )
    train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=2, shuffle=False, num_workers=0)
    model = Unet(in_channels=3, out_channels=1).to(device)
    criterion = combined_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)