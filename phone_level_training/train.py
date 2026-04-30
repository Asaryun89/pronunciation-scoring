import torch
from datasets import load_dataset

from .model import PhoneModel
from .collator import collate_fn
from .data import PhoneDataset


def train():

    dataset = load_dataset("mispeech/speechocean762")

    model = PhoneModel().cuda()

    loader = torch.utils.data.DataLoader(
        PhoneDataset(dataset["train"], ...),  # fill model + processor + phone2id
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(20):
        for x, y, mask in loader:
            x = x.cuda()
            y = y.cuda()
            mask = mask.cuda()

            pred = model(x, mask)

            loss = ((pred - y)**2 * mask).sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("epoch", epoch, "loss", loss.item())


if __name__ == "__main__":
    train()