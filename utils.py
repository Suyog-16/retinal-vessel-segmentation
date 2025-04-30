import torch
def dice_score(y_pred,y_true,smooth= 1e-15):
    """
    Computes the Dice coefficient for binary segmeantaion

    Args:
    - y_pred : torch tensor,predicted binary mask(0 or 1)
    - y : torch tensor, ground truth binary mask(0 or 1)
    - smooth : small number to avoid divison by zero

    Returns:
    - Dice coefficient (float)
    """
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    intersection = torch.sum(y_true * y_pred)
    total = torch.sum(y_true) + torch.sum(y_pred)

    dice = (2. * intersection + smooth) / (total + smooth)
    return dice.item()

