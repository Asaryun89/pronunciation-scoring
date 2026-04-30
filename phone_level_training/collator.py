import torch

def collate_fn(batch):
    feats, scores = zip(*batch)

    max_len = max(len(f) for f in feats)
    dim = feats[0].shape[1]

    x = torch.zeros(len(batch), max_len, dim)
    y = torch.zeros(len(batch), max_len)
    mask = torch.zeros(len(batch), max_len)

    for i in range(len(batch)):
        l = len(feats[i])
        x[i, :l] = feats[i]
        y[i, :l] = scores[i]
        mask[i, :l] = 1

    return x, y, mask