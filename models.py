import torch
from pos_enc import PositionalEncoding
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, vocab_size):
        super(Baseline, self).__init__()

        self.emb = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.emb(idx) # B ,T , C
        # targets are of shape # B, T

        if targets == None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

        return logits, loss
    

    
    def generate(self, idx, lenght):
        # idx is (B, T)
        for _ in range(lenght):
            logits, loss = self.forward(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            id_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, id_next], dim=-1)
        return idx




    
