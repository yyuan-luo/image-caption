import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, embedding_size, feature_size, output_size, hidden_dim=12, num_layers=1):
        super(Decoder, self).__init__()
        self.num_embeddings = embedding_size
        self.feature_size = feature_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(self.feature_size, self.hidden_dim, self.num_layers, batch_first=True)
        self.embedding = nn.Embedding(self.num_embeddings, self.feature_size)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, captions):
        batch_size = captions.shape[0]
        hidden_0 = self.init_hidden(batch_size)  # TODO: replace this with image features from encoder

        embedding_captions = self.embedding(captions)
        out, hidden = self.rnn(embedding_captions, hidden_0)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return hidden
