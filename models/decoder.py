import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, output_size, hidden_dim=512, num_layers=1, device=torch.device("cpu")):
        super(Decoder, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.rnn = nn.RNN(self.embedding_size, self.hidden_dim, self.num_layers, batch_first=True)
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, image_features, captions):
        batch_size = captions.shape[0]
        hidden_0 = self.init_hidden(batch_size)
        hidden_0 = hidden_0.to(self.device)
        captions = captions[:, :-1]     # TODO: this might be wrong, the goal is to remove '[EOS]' token
        embedding_captions = self.embedding(captions)
        embedding_captions = torch.cat((image_features.unsqueeze(1), embedding_captions), 1)    # Pre-inject
        out, hidden = self.rnn(embedding_captions, hidden_0)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return hidden

    def sampler(self, image_features, states=None, max_len=20):
        sent_out = []
        input = image_features
        for i in range(max_len):
            out, _ = self.rnn(input, states)
            out = self.fc(out.contiguous().view(-1, self.hidden_dim))
            best = out.max(1)[1]    # tensor.out => (value, index)
            sent_out.append(best.item())
            input = self.embedding(best).unsqueeze(1)
        return sent_out