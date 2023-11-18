import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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

    def forward(self, image_features, captions, seq_lens):
        batch_size = captions.shape[0]
        hidden_0 = self.init_hidden(batch_size)
        hidden_0 = hidden_0.to(self.device)
        captions = captions[:, :-1]     # TODO: this might be wrong, the goal is to remove '[EOS]' token
        sorted_lens, sorted_indices = seq_lens.sort(descending=True)
        embedding_captions = self.embedding(captions)
        embedding_captions = torch.cat((image_features.unsqueeze(1), embedding_captions), 1)    # Pre-inject'
        sorted_embedding_captions = embedding_captions[sorted_indices]
        packed_input = pack_padded_sequence(sorted_embedding_captions, sorted_lens.cpu().numpy(), batch_first=True)
        packed_out, hidden = self.rnn(packed_input, hidden_0)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        _, original_indices = sorted_indices.sort()
        out = out[original_indices]
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return hidden

    def sampler(self, image_features, states=None, max_len=15):
        sent_out = []
        input = image_features
        input = input.unsqueeze(0)
        for i in range(max_len):
            out, _ = self.rnn(input, states)
            input = out
            out = self.fc(out.contiguous().view(-1, self.hidden_dim))
            best = out.max(1)[1]    # tensor.out => (value, index)
            sent_out.append(best.item())
        return sent_out