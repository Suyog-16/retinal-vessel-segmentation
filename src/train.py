import torch
from tqdm import tqdm
from torch import nn
from dataset import DRIVE_dataset
from utils import dice_loss,dice_score,combined_loss,iou_score
from torchvision import transforms as T
from torch.utils.data import DataLoader
from timeit import default_timer as timer
import albumentations as A
from albumentations import ToTensorV2
from model import Unet
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Training script")
parser.add_argument('--batch_size',type=int,default=2,required=False,help="Batch_size of model")
parser.add_argument('--lr',type=float,default=0.0001,required=False,help="learning rate of model")
parser.add_argument('--epochs',type=int,default=50,required=False,help="No. of epochs")

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else "cpu"



def train_step(model, dataloader, criterion, optimizer):
    """
    Performs one training step over a single epoch

    Args:
    - model(torch.nn.Module) : The model to be trained
    - dataloader(torch.utils.data.DataLoader) : Dataloader for training data
    - criterion(torch.nn.Module) : Loss function to calculate loss
    - optimizer(torch.optim.Optimizer) : Optimizer to update the model parameters


    Returns:
    - tuple : A tuple containing:
        - train_loss(float) : Average training loss over the training samples
        - train_dice(float) : Average Dice coefficient score over the training samples
        - train_iou(flaot) : Average Intersection over union score over the training samples

    
    """
    model.train()
    train_loss, train_dice,train_iou = 0, 0,0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_dice += dice_score(y_pred, y)
        train_iou += iou_score(y_pred,y)
    train_loss /= len(dataloader)
    train_dice /= len(dataloader)
    train_iou /= len(dataloader)
    return train_loss, train_dice,train_iou



def test_step(model, dataloader, criterion):
    """
    Performs one testing step over a single epoch

    Args:
    - model(torch.nn.Module) : The model to be tested
    - dataloader(torch.utils.data.DataLoader) : Dataloader for testing data
    - criterion(torch.nn.Module) : Loss function to calculate loss
    


    Returns:
    - tuple : A tuple containing:
        - test_loss(float) : Average testing loss over the testing samples
        - test_dice(float) : Average Dice coefficient score over the testing samples
        - test_iou(flaot) : Average Intersection over union score over the testing samples
    """
    model.eval()
    test_loss, test_dice,test_iou = 0, 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            test_loss += loss.item()
            test_dice += dice_score(y_pred, y)
            test_iou += iou_score(y_pred,y)
            #if batch == 8 :
                #utils.save_image(pred, f"pred_epoch_{epoch+1}.png")
                #utils.save_image(y, f"target_epoch_{epoch+1}.png")
            if batch == 0:
                pred = torch.sigmoid(y_pred)
                pred = (pred > 0.5).float()

    test_loss /= len(dataloader)
    test_dice /= len(dataloader)
    test_iou /= len(dataloader)
    return test_loss, test_dice,test_iou

def train(model, train_dataloader, test_dataloader, optimizer, criterion, epochs=100):

    results = {"train_loss": [], "train_dice": [],"train_iou": [], "test_loss": [], "test_dice": [],"test_iou":[]}
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    for epoch in tqdm(range(epochs)):
        train_loss, train_dice, train_iou = train_step(model, train_dataloader, criterion, optimizer)
        test_loss, test_dice,test_iou = test_step(model, test_dataloader, criterion, epoch)
        print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_dice: {train_dice:.4f} | train_iou : {train_iou:.4f} |"
              f"test_loss: {test_loss:.4f} | test_dice: {test_dice:.4f} | test_iou : {test_iou:.4f}")
        #scheduler.step(test_loss)
        results["train_loss"].append(train_loss)
        results["train_dice"].append(train_dice)
        results["train_iou"].append(train_iou)
        results["test_loss"].append(test_loss)
        results["test_dice"].append(test_dice)
        results["test_iou"].append(test_iou)
    return results


if __name__ == "__main__":
    transform = A.Compose([
    A.Resize(512, 512),
    #A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    #A.RandomBrightnessContrast(p=0.3),
    #A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=10, p=0.5),
    A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
    ToTensorV2(),
    ])
   
    train_data = DRIVE_dataset(
        img_dir="C:/Users/Acer nitro/Desktop/retinal-vessel-segmentation/data/DRIVE/training/images",
        mask_dir="C:/Users/Acer nitro/Desktop/retinal-vessel-segmentation/data/DRIVE/training/1st_manual",
        transform=transform
    )
    test_data = DRIVE_dataset(
        img_dir="C:/Users/Acer nitro/Desktop/retinal-vessel-segmentation/data/DRIVE/test/images",
        mask_dir="C:/Users/Acer nitro/Desktop/retinal-vessel-segmentation/data/DRIVE/test/1st_manual",
        transform=transform
    )
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
   

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    model = Unet(in_channels=3, out_channels=1).to(device)
    criterion = combined_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

start_time = timer()
results = train(model, train_dataloader, test_dataloader, optimizer, criterion, epochs=EPOCHS)
end_time = timer()
print(f"Total training time: {end_time - start_time:.3f} seconds")