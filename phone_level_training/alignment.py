import torch


def align(log_probs, phone_ids):
    """
    Assign audio frames to phone-sequence positions using even splits.

    Even splits are used because the available HuBERT checkpoint was trained
    for character-level ASR (not phoneme CTC), so its output vocabulary does
    not correspond to phoneme IDs.  CTC forced-alignment would require a model
    fine-tuned on a phoneme target sequence (e.g. Arpabet).

    Args:
        log_probs:  (T_frame, vocab)  — unused here, kept for API compatibility
        phone_ids:  list[int]         — target phone IDs in sequence order

    Returns:
        frame2phone: (T_frame,) long tensor
            frame2phone[t] = i  means frame t is assigned to phone i
    """
    T = log_probs.shape[0]
    N = len(phone_ids)

    if T >= N:
        # Normal case: every phone gets at least one frame.
        frame2phone = torch.zeros(T, dtype=torch.long)
        for i in range(N):
            start = i * T // N
            end   = (i + 1) * T // N if i < N - 1 else T
            frame2phone[start:end] = i
    else:
        # Fewer frames than phones: assign each frame to the nearest phone,
        # leaving some phones empty (handled by zeros in aggregate_ssl).
        frame2phone = torch.arange(T, dtype=torch.float)
        frame2phone = (frame2phone * N / T).long().clamp(0, N - 1)

    return frame2phone
