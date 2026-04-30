import torch

def extract_ssl_and_logprob(model, input_values):
    """
    Input:
        input_values: (B, T)

    Output:
        ssl: (T_frame, 1024)
        log_probs: (T_frame, vocab)
    """
    with torch.no_grad():
        out = model(
            input_values,
            output_hidden_states=True
        )

        ssl = out.hidden_states[-1].squeeze(0)     # (T, 1024)
        logits = out.logits.squeeze(0)
        log_probs = torch.log_softmax(logits, dim=-1)

    return ssl, log_probs

def aggregate_to_phone(ssl, frame2phone, num_phones):
    out = []
    for i in range(num_phones):
        idx = (frame2phone == i).nonzero().squeeze(-1)
        out.append(ssl[idx].mean(0))
    return torch.stack(out)


def compute_duration(frame2phone, num_phones):
    dur = []
    for i in range(num_phones):
        dur.append((frame2phone == i).sum())
    dur = torch.tensor(dur, dtype=torch.float)
    return torch.log(dur + 1).unsqueeze(-1)


def compute_gop(log_probs, phone_ids, frame2phone):
    gop = []

    for i, p in enumerate(phone_ids):
        idx = (frame2phone == i).nonzero().squeeze(-1)
        lp = log_probs[idx]

        target = lp[:, p].mean()

        other = torch.cat([lp[:, :p], lp[:, p+1:]], dim=-1)
        max_other = other.max(dim=-1)[0].mean()

        gop.append(target - max_other)

    gop = torch.stack(gop)
    gop = (gop - gop.mean()) / (gop.std() + 1e-5)

    return gop.unsqueeze(-1)