import torch


def batch_dice_iou_from_logits(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()

    dims = (1, 2, 3)
    intersection = torch.sum(preds * targets, dim=dims)
    pred_sum = torch.sum(preds, dim=dims)
    target_sum = torch.sum(targets, dim=dims)
    union = pred_sum + target_sum - intersection

    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()


def dice_coef(y_true, y_pred, thr=0.5, epsilon=1e-7):
    if y_pred.shape[1] != 1:
        y_pred = y_pred[:, 1:2]
    dice, _ = batch_dice_iou_from_logits(y_pred, y_true, threshold=thr, eps=epsilon)
    return dice


def calculate_iou(y_true, y_pred, thr=0.5, epsilon=1e-7):
    if y_pred.shape[1] != 1:
        y_pred = y_pred[:, 1:2]
    _, iou = batch_dice_iou_from_logits(y_pred, y_true, threshold=thr, eps=epsilon)
    return iou
