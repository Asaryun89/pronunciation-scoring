def fuse_scores(utt_score_0_100: float, mean_word_0_1: float, prosody_0_1: float,
                alpha: float = 0.6, beta: float = 0.25, gamma: float = 0.15) -> float:
    """
    Combine multiple dimensions into final 0-100.
    """
    acc_0_100 = mean_word_0_1 * 100.0
    pro_0_100 = prosody_0_1 * 100.0
    final = alpha * acc_0_100 + beta * utt_score_0_100 + gamma * pro_0_100
    # clamp
    return max(0.0, min(100.0, float(final)))