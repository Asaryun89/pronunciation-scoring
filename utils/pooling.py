import numpy as np

def mean_pool(emb: np.ndarray, i0: int, i1: int) -> np.ndarray:
    seg = emb[i0:i1]
    if seg.size == 0:
        return np.zeros((emb.shape[1],), dtype=np.float32)
    return seg.mean(axis=0).astype(np.float32)

def utt_pool_mean_std(emb: np.ndarray) -> np.ndarray:
    """
    Global pooling: mean + std concatenation.
    Output dim = 2D
    """
    mu = emb.mean(axis=0)
    sd = emb.std(axis=0)
    return np.concatenate([mu, sd], axis=0).astype(np.float32)