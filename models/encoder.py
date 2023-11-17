import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # resnet50 as backbone
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        modules = list(self.resnet50.children())[:-2]
        self.resnet50 = nn.Sequential(*modules)

    def forward(self, image):
        features = self.resnet50(image)

        return features
