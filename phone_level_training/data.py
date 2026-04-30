import torch
from alignment import align
from features import extract_ssl_and_logprob, aggregate_to_phone, compute_gop, compute_duration

class PhoneDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, model, processor, phone2id):
        self.dataset = dataset
        self.model = model
        self.processor = processor
        self.phone2id = phone2id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        audio = sample["audio"]["array"]

        phones = []
        scores = []

        for w in sample["words"]:
            phones.extend(w["phones"])
            scores.extend(w["phones-accuracy"])

        phone_ids = [self.phone2id[p] for p in phones]

        input_values = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_values.cuda()

        ssl, log_probs = extract_ssl_and_logprob(self.model, input_values)

        frame2phone = align(log_probs, phone_ids)

        phone_ssl = aggregate_to_phone(ssl, frame2phone, len(phone_ids))
        gop = compute_gop(log_probs, phone_ids, frame2phone)
        dur = compute_duration(frame2phone, len(phone_ids))

        phone_ids_tensor = torch.tensor(phone_ids)
        phone_embed = self.embedding(phone_ids_tensor)

        feat = torch.cat([phone_ssl, gop, dur, phone_embed], dim=-1)

        return feat, torch.tensor(scores, dtype=torch.float)