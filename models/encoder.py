import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # Adjust the image size to 224 x 224
        self.resize = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=54, stride=2)

        # resnet50 as backbone
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        modules = list(self.resnet50.children())[:-2]
        self.resnet50 = nn.Sequential(*modules)

    def forward(self, image):
        x = self.resize(image)
        features = self.resnet50(torch.unsqueeze(x, 0))

        return features
