import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class Encoder(nn.Module):
    def __init__(self, embedding_size):
        super(Encoder, self).__init__()
        # resnet50 as backbone
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        modules = list(self.resnet50.children())[:-1]
        self.resnet50 = nn.Sequential(*modules)
        self.embedding = nn.Linear(2048, embedding_size)    # TODO: replace 2048 with a variable

    def forward(self, image):
        features = self.resnet50(image)
        features = features.view(features.size(0), -1)
        features = self.embedding(features)
        return features
