import torch

def extract_ssl_and_logprob(model, input_values):
    """
    Input:
        input_values: (1, T_samples)

    Output:
        ssl:       (num_layers, T_frame, 1024)  — all transformer layers (CNN layer excluded)
        log_probs: (T_frame, vocab)
    """
    with torch.no_grad():
        out = model(input_values, output_hidden_states=True)
        # hidden_states[0] is the CNN feature projection (512-dim); skip it.
        # hidden_states[1:] are the transformer layers (1024-dim each).
        ssl = torch.stack([h.squeeze(0) for h in out.hidden_states[1:]], dim=0)
        logits = out.logits.squeeze(0)
        log_probs = torch.log_softmax(logits, dim=-1)

    ssl = ssl.nan_to_num(0.0)
    return ssl, log_probs


def aggregate_ssl(ssl, frame2phone, num_phones):
    """
    Mean-pool frame-level SSL features into phone-level features.

    Args:
        ssl:         (num_layers, T_frame, 1024)
        frame2phone: (T_frame,) — frame-to-phone index mapping
        num_phones:  int

    Returns:
        (num_phones, num_layers, 1024)
    """
    frame2phone = frame2phone.to(ssl.device)
    num_layers, _, ssl_dim = ssl.shape
    out = []
    for i in range(num_phones):
        idx = (frame2phone == i).nonzero().squeeze(-1)
        if idx.numel() == 0:
            # Phone received no frames (T < N edge case) — use zero vector.
            out.append(torch.zeros(num_layers, ssl_dim, device=ssl.device))
        else:
            out.append(ssl[:, idx, :].mean(dim=1))  # (num_layers, ssl_dim)
    return torch.stack(out)  # (num_phones, num_layers, ssl_dim)


def compute_duration(frame2phone, num_phones):
    dur = []
    for i in range(num_phones):
        dur.append((frame2phone == i).sum())
    dur = torch.tensor(dur, dtype=torch.float)
    return torch.log(dur + 1).unsqueeze(-1)


def compute_gop(log_probs, phone_ids, frame2phone):
    """
    Returns zeros — GOP requires a phoneme-level CTC model whose output
    vocabulary aligns with phone_ids.  The current checkpoint is character-level
    ASR, so its log-probs do not correspond to phoneme IDs.  Replace this
    checkpoint with a phoneme CTC model (e.g. wav2vec2 fine-tuned on Arpabet)
    to get meaningful GOP scores.
    """
    return torch.zeros(len(phone_ids), 1)