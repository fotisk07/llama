from pos_enc import PositionalEncoding, Embeddings
import matplotlib.pyplot as plt
import torch
from utils import encode, get_batches, decode

lines = open("input.txt", 'r').read()

vocab = sorted(list(set(lines)))
itos = {i:ch for i, ch in enumerate(vocab)}
stoi = {ch:i for i, ch in enumerate(vocab)}


dataset = torch.tensor(encode(lines, stoi))
xs, ys = get_batches(dataset, 'train', 5 ,14)

print(xs.shape)

