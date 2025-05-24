import torch
from torch import nn
device = 'cuda' if torch.cuda.is_available() else "cpu"


def dice_score(pred, target, smooth=1e-6):
    """
    Calculates the dice coeficient between target and predicted tensors

    Args:
    - pred(torch.tensor) : The predicted output tensor
    - target(torch.tensor) : Ground truth binary tensor
    - smooth(float) : A small constant to avoid division by zero error


    Returns:

    Dice coefficient score between 0.0 to 1.0

    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    target = target.float()
    pred_sum = pred.sum().item()
    target_sum = target.sum().item()
    #print(f"Pred sum: {pred_sum}, Target sum: {target_sum}")
    intersection = (pred * target).sum()
    denominator = pred.sum() + target.sum() + smooth
    if denominator == smooth:
        return 0.0
    return (2. * intersection + smooth) / denominator

def dice_loss(pred, target, smooth=1e-6):
    """
    Computes the Dice loss for the given target and predicted tensors

    Args:
    - pred(torch.tensor) : The predicted output tensor
    - target(torch.tensor) : Ground truth binary tensor
    - smooth(float) : A small constant to avoid division by zero error

    Returns:
    The Dice loss for backpropagation


    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    target = target.float()
    intersection = (pred * target).sum()
    denominator = pred.sum() + target.sum() + smooth
    if denominator == smooth:
        return 1.0
    return 1 - ((2. * intersection + smooth) / denominator)

def combined_loss(y_pred, y_true):
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).to(device))(y_pred, y_true)
    dice = dice_loss(y_pred, y_true)
    return 0.5 * bce + 0.5 * dice

def iou_score(pred, target, smooth=1e-6):
    """
    Computes the Intersection over union score between predicted and target tensors

    Args:
    - pred(torch.tensor) : The predicted output tensor
    - target(torch.tensor) : Ground truth binary tensor
    - smooth(float) : A small constant to avoid division by zero error

    Returns:
    Intersection over union score between 0.0 to 1.0

    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    if union == 0:
        return 0.0
    return (intersection + smooth) / (union + smooth)