def masked_mse(pred, target, mask):
    diff = (pred - target) ** 2
    diff = diff * mask
    return diff.sum() / mask.sum()