import torch
import torchaudio

def align(log_probs, phone_ids):
    """
    log_probs: (T_frame, vocab)
    phone_ids: (T_phone)

    return:
        frame2phone: (T_frame,)
    """
    log_probs = log_probs.cpu()
    targets = torch.tensor(phone_ids, dtype=torch.int32)

    # torchaudio expects (vocab, time)
    emissions = log_probs.transpose(0, 1)

    alignment = torchaudio.functional.forced_align(
        emissions,
        targets
    )

    frame2phone = alignment[0]  # depends on torchaudio version

    return frame2phone