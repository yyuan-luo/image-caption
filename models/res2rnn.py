import torch
from torch import nn

from models.encoder import Encoder
from models.decoder import Decoder


class Res2RNN(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, hidden_dim=256, num_layers=1):
        super(Res2RNN, self).__init__()
        self.encoder = Encoder(embedding_size)
        self.decoder = Decoder(vocabulary_size, embedding_size, hidden_dim, num_layers)

    def forward(self, images, captions, seq_lengths):
        image_feature = self.encoder(images)
        pre_captions = self.decoder(image_feature, captions, seq_lengths)
        return pre_captions


if __name__ == '__main__':
    embedding_size = 256
    images = torch.randn(3, 3, 224, 224)
    captions = torch.randint(1, 10, [3, 20])
    seq_lengths = torch.tensor([10, 12, 20], dtype=torch.int)
    print(images.shape, captions.shape, seq_lengths.shape)
    model = Res2RNN(vocabulary_size=100, embedding_size=embedding_size)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore <PAD>
    output = model(images, captions, seq_lengths)
    print(output.view(-1, 100).shape, captions.view(-1).shape)
    print(criterion(output.view(-1, 100), captions.view(-1)))
