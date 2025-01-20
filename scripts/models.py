import torch.nn as nn
from torchvision import models

class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=False, num_classes=1)
    
    def forward(self, x):
        return nn.Sigmoid()(self.model(x))

class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        self.model = models.googlenet(pretrained=False, num_classes=1)
    
    def forward(self, x):
        if self.training:
            return nn.Sigmoid()(self.model(x).logits)
        else:
            return nn.Sigmoid()(self.model(x))

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = models.resnet50(pretrained=False, num_classes=1)
    
    def forward(self, x):
        return nn.Sigmoid()(self.model(x))
    
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.model = models.alexnet(pretrained=False, num_classes=1)
    
    def forward(self, x):
        return nn.Sigmoid()(self.model(x))

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.model = models.vgg19_bn(pretrained=False, num_classes=1)
    
    def forward(self, x):
        return nn.Sigmoid()(self.model(x))