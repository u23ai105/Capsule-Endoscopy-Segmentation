import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for image semantic segmentation task.
    Sørensen's original formula was intended to be applied to discrete data.
    Given two sets, X and Y, it is defined as
    {\displaystyle DSC={\frac {2|X\cap Y|}{|X|+|Y|}}}
    where |X| and |Y| are the cardinalities of the two sets
    (i.e. the number of elements in each set).
    The Sørensen index equals twice the number of elements common
    to both sets divided by the sum of the number of elements in each set.
    Reference:
    https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Parameters
    ----------
    from_logits: boolean, default=False
        If True, assumes input is raw logits.
    smooth: float, default=1e-7
        Smoothness constant for dice coefficient.
    eps: float, default=1e-7
        A small epsilon for numerical stability to avoid zero division error
        (denominator will be always greater or equal to eps).
    reduction: string, default=None
        Reduction method to apply, return mean over batch if 'mean',
        return sum if 'sum', return a tensor of shape [N,] if None.
    Shape:
        - **y_pred** - torch.Tensor of shape (N, C, H, W)
        - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)
    Returns
    -------
    loss: torch.tensor
        Calculated dice loss.
    """

    def __init__(self, from_logits=False, smooth=1e-7, eps=1e-7, reduction=None):
        super(DiceLoss, self).__init__()
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        if self.from_logits: y_pred = F.sigmoid(y_pred)
        # flatten label and prediction tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = torch.sum(y_pred * y_true)
        cardinalities = torch.sum(y_pred + y_true) + self.smooth
        dice = (2.0 * intersection + self.smooth) / (cardinalities + self.smooth).clamp_min(self.eps)
        loss = 1 - dice
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction is None:
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceBCELoss(nn.Module):
    """
    A compound loss function that combines Dice Loss and Binary Cross-Entropy.
    This can lead to more stable training and better performance.
    """

    def __init__(self, weight_dice=0.5, weight_bce=0.5):
        super(DiceBCELoss, self).__init__()
        self.dice_loss = DiceLoss(from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        return (self.weight_dice * dice) + (self.weight_bce * bce)


class FocalLoss(nn.Module):
    """
    Focal Loss, designed to address class imbalance by focusing on hard-to-classify examples.
    """

    def __init__(self, alpha=0.25, gamma=2.0, from_logits=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def forward(self, inputs, targets):
        if self.from_logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        p_t = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss

        return focal_loss.mean()
# More losses coming soon...