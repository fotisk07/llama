import torch
import torch.nn as nn
from utils import get_batches


def train_model(model, dataset, lr, epochs, batch_size, context_window):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for i in range(epochs):
        ix = torch.randint(0, dataset.size(0) - context_window - 1, (batch_size,))
        xs = torch.stack([dataset[i:i+context_window] for i in ix]).long()
        ys = torch.stack([dataset[i+1:i+context_window+1] for i in ix]).long()

        optimizer.zero_grad()
        
        logits, loss = model(xs, ys)
        loss.backward()
        optimizer.step()
        print(f"Epoch {i} loss: {loss.item()}")

    return model