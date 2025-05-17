# utils.py

# import torch


def dice_loss(pred, target):
    smooth = 1.0
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
