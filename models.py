import torch
from pos_enc import PositionalEncoding
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, vocab_size, emb_size, head_size, context_window):
        super(Baseline, self).__init__()

        self.emb = nn.Embedding(vocab_size, emb_size)
        
        self.posemb = PositionalEncoding(emb_size, context_window, vocab_size=vocab_size)

        self.attention = MultiSelfAttention(head_size= emb_size // 4, emb_size=emb_size, context_window=context_window, n_heads=4)
        
        self.lm_head = nn.Linear(emb_size, vocab_size)

        self.context_window = context_window    


    def forward(self, idx, targets=None):
        x = self.emb(idx) # B ,T , C
        pos_encodings = self.posemb(x)

        x = x + pos_encodings

        x = self.attention(x)


        logits = self.lm_head(x)


        # targets are of shape # B, T

        if targets == None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

        return logits, loss
    
    def generate(self, idx, lenght):
        # idx is (B, T)
        self.eval()
        for _ in range(lenght):
            logits, loss = self.forward(idx[:, -self.context_window:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            id_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, id_next], dim=-1)
        return idx


class SingleHeadAttention(nn.Module):
    def __init__(self, head_size, emb_size, context_window):
        super(SingleHeadAttention, self).__init__()

        self.head_size = head_size
        self.q = nn.Linear(emb_size, head_size, bias=False)
        self.k = nn.Linear(emb_size, head_size, bias=False)
        self.v = nn.Linear(emb_size, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones((context_window, context_window))))   

    def forward(self, x):
        B,T,C = x.shape
        q = self.q(x)
        k = self.k(x)

        # Calculate affinities
        wei = q @ k.transpose(1,2) / (C ** 0.5)
        wei = wei.masked_fill(self.tril[:T,:T] == 0 , float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.v(x)
        out = wei @ v

        return out



class MultiSelfAttention(nn.Module):
    def __init__(self, head_size, emb_size, context_window, n_heads):
        super(MultiSelfAttention, self).__init__()
        self.attentions = nn.ModuleList([SingleHeadAttention(head_size, emb_size, context_window) for _ in range(n_heads)])

    def forward(self, x):
        return torch.cat([att(x) for att in self.attentions], dim=-1)


