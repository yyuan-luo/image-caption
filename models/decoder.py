import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, output_size, hidden_dim=512, num_layers=1):
        super(Decoder, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(self.embedding_size, self.hidden_dim, self.num_layers, batch_first=True)
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, image_features, captions):
        batch_size = captions.shape[0]
        hidden_0 = self.init_hidden(batch_size)
        captions = captions[:, :-1]
        embedding_captions = self.embedding(captions)
        print(image_features.unsqueeze(1).shape, embedding_captions.shape)
        embedding_captions = torch.cat((image_features.unsqueeze(1), embedding_captions), 1)    # Pre-inject
        out, hidden = self.rnn(embedding_captions, hidden_0)
        print(out.shape)
        out = out.contiguous().view(-1, self.hidden_dim)
        print(out.shape)
        out = self.fc(out)

        return out

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return hidden
