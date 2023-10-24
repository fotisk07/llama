import torch


lines = open("input.txt", 'r').read()

vocab = sorted(list(set(lines)))
itos = {i:ch for i, ch in enumerate(vocab)}
stoi = {ch:i for i, ch in enumerate(vocab)}

def encode(s):
    return [stoi[ch] for ch in s]

def decode(l):
    return ''.join([itos[i] for i in l])


MASTER_CONFIG = {
    "vocab_size" : len(vocab),
}

MASTER_CONFIG.update({
    "batch_size" : 8
})

dataset = torch.tensor(encode(lines))

def get_batches(data, split, batch_size,
                context_window, config=MASTER_CONFIG):
    train = data[:int(.8*len(data))]
    val = data[int(.8 * len(data)) : int(.9 * len(data))]
    test = data[int(.9 * len(data))]

    batch_data = train
    if split == "val":
        batch_data = val
    if split == "test":
        batch_data = test

    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()

    return x, y


MASTER_CONFIG.update({
    'batch_size': 8,
    'context_window': 16
})

xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

[(decode(xs[i].tolist()), decode(ys[i].tolist())) for i in range(len(xs))]

