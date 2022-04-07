import torch
import torch.nn as nn
from torchvision import models

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        net = models.resnet50(pretrained=False)
        channel_in = net.fc.in_features
        net.fc = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(
            nn.Linear(channel_in, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x