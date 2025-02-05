import torch.nn as nn
from torchvision import models

USE_PRETRAINED = True


class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=USE_PRETRAINED)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))


class GoogleNet(nn.Module):
    def __init__(self, aux_logits=False):
        super(GoogleNet, self).__init__()
        self.model = models.googlenet(pretrained=USE_PRETRAINED, aux_logits=aux_logits)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        self.sigmoid = nn.Sigmoid()
        self.aux_logits = aux_logits

    def forward(self, x):
        x = self.model(x)
        if self.aux_logits:  # Handle auxiliary outputs if enabled
            x = x[0]  # Use main classifier output
        return self.sigmoid(x)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = models.resnet50(pretrained=USE_PRETRAINED)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.model = models.alexnet(pretrained=USE_PRETRAINED)
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.model = models.vgg19_bn(pretrained=USE_PRETRAINED)
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))
