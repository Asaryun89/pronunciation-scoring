import torch

def collate_fn(batch):
    """
    batch: list of dict
        {
            "ssl": (T, 1024)
            "gop": (T, 1)
            "dur": (T, 1)
            "phone_ids": (T,)
            "scores": (T,)
        }
    """

    B = len(batch)
    max_len = max(item["ssl"].shape[0] for item in batch)

    ssl_dim = batch[0]["ssl"].shape[1]

    # ===== allocate =====
    ssl = torch.zeros(B, max_len, ssl_dim)
    gop = torch.zeros(B, max_len, 1)
    dur = torch.zeros(B, max_len, 1)
    phone_ids = torch.zeros(B, max_len, dtype=torch.long)
    scores = torch.zeros(B, max_len)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    # ===== fill =====
    for i, item in enumerate(batch):
        T = item["ssl"].shape[0]

        ssl[i, :T] = item["ssl"]
        gop[i, :T] = item["gop"]
        dur[i, :T] = item["dur"]
        phone_ids[i, :T] = item["phone_ids"]
        scores[i, :T] = item["scores"]

        mask[i, :T] = 1

    return {
        "ssl": ssl,
        "gop": gop,
        "dur": dur,
        "phone_ids": phone_ids,
        "scores": scores,
        "mask": mask
    }