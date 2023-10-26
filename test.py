from pos_enc import PositionalEncoding
import matplotlib.pyplot as plt
import torch
from utils import encode, get_batches, decode
from models import Baseline


# Data stuff
lines = open("input.txt", 'r').read()

vocab = sorted(list(set(lines)))
itos = {i:ch for i, ch in enumerate(vocab)}
stoi = {ch:i for i, ch in enumerate(vocab)}
vocab_size = len(vocab)

dataset = torch.tensor(encode(lines, stoi))
xs, ys = get_batches(dataset, 'train', 5 ,14)

# Definition
nn = Baseline(vocab_size)

# Training


# Generation
idx = torch.zeros((1,1), dtype=torch.long)
print(decode(nn.generate(idx, 100)[0].tolist(), itos))
