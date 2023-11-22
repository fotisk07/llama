from pos_enc import PositionalEncoding
import matplotlib.pyplot as plt
import torch
from utils import encode, decode
from trainer import train_model
from models import Baseline


# Data stuff
lines = open("input.txt", 'r').read()

# Tokenization utils
vocab = sorted(list(set(lines)))
itos = {i:ch for i, ch in enumerate(vocab)}
stoi = {ch:i for i, ch in enumerate(vocab)}
vocab_size = len(vocab)

# Tokenize
dataset = torch.tensor(encode(lines, stoi))
train = dataset[:int(.8*len(dataset))]
val = dataset[int(.8 * len(dataset)) : int(.9 * len(dataset))]
test = dataset[int(.9 * len(dataset))]


# Define model
model = Baseline(vocab_size)

# Train
context_window = 10
batch_size = 32
print("Training...")
model = train_model(model, train, lr=1e-3, epochs=10, batch_size=batch_size, context_window=context_window)


# Generate
print("\nGenerating...")
idx = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(idx, 100)[0].tolist(), itos))