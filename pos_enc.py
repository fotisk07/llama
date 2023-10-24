import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_lenght, vocab_size):
        super(PositionalEncoding, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)

        self.max_seq_lenght = max_seq_lenght
        self.d_model = d_model
        self.vocab_size = vocab_size

        pe = torch.zeros((self.max_seq_lenght , self.d_model))

        pos = torch.arange(self.max_seq_lenght).unsqueeze(1)
        helper = torch.pow(10000, torch.arange(0,self.d_model,2) / self.d_model)

        pe[:,0::2] = torch.sin(pos / helper)
        pe[:, 1::2] = torch.cos(pos/ helper )

        self.register_buffer('pe', pe)


    def forward(self, x):
        # x is of size [N, seq_lenght, d_model]
        seq_length= x.shape[1]
        return x + self.pe[:seq_length]
    


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, input):
        #input [N, seq_lenght]

        input = self.emb(input)

        return input