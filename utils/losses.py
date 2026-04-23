import torch
import torch.nn.functional as F


def dice_loss_from_logits(logits, targets, smooth=1e-5):
    probs = torch.sigmoid(logits)
    return dice_loss_from_probs(probs, targets, smooth=smooth)


def dice_loss_from_probs(probs, targets, smooth=1e-5):
    targets = targets.float()
    probs = probs.float()
    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def supervised_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets.float())
    dice = dice_loss_from_logits(logits, targets)
    return bce + dice


def masked_bce_loss(logits, targets, valid_mask):
    loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    valid_mask = valid_mask.float()
    return (loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)


def masked_dice_loss(logits, targets, valid_mask, smooth=1e-5):
    probs = torch.sigmoid(logits)
    targets = targets.float()
    valid_mask = valid_mask.float()
    intersection = torch.sum(probs * targets * valid_mask, dim=(1, 2, 3))
    denominator = torch.sum((probs + targets) * valid_mask, dim=(1, 2, 3))
    valid_images = torch.sum(valid_mask, dim=(1, 2, 3)) > 0
    if not torch.any(valid_images):
        return logits.new_tensor(0.0)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice[valid_images].mean()


def consistency_loss(logits, pseudo_targets, valid_mask):
    if valid_mask.sum() == 0:
        return logits.new_tensor(0.0)
    return masked_bce_loss(logits, pseudo_targets, valid_mask) + masked_dice_loss(
        logits, pseudo_targets, valid_mask
    )
