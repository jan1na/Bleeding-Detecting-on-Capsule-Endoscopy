import torch.nn as nn
from torchvision import models
import torch

# Flag to determine whether to use pretrained weights for the models
USE_PRETRAINED = True


class MobileNetV2(nn.Module):
    """
    MobileNetV2 model for binary classification of images.
    Uses a pretrained MobileNetV2 model with a modified classifier for single output.
    """

    def __init__(self):
        """
        Initialize the MobileNetV2 model with a modified classifier layer.

        The classifier layer is changed to output a single value, making it suitable for binary classification.
        """
        super(MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=USE_PRETRAINED)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MobileNetV2 model.

        :param x: Input tensor of shape (batch_size, channels, height, width)
        :return: Sigmoid activated output tensor representing probability of the positive class
        """
        return self.sigmoid(self.model(x))


class ResNet(nn.Module):
    """
    ResNet model for binary classification of images.
    Uses a pretrained ResNet50 model with a modified fully connected layer.
    """

    def __init__(self):
        """
        Initialize the ResNet model with a modified fully connected layer.

        The final fully connected layer is adjusted to output a single value for binary classification.
        """
        super(ResNet, self).__init__()
        self.model = models.resnet50(pretrained=USE_PRETRAINED)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ResNet model.

        :param x: Input tensor of shape (batch_size, channels, height, width)
        :return: Sigmoid activated output tensor representing probability of the positive class
        """
        return self.sigmoid(self.model(x))


class AlexNet(nn.Module):
    """
    AlexNet model for binary classification of images.
    Uses a pretrained AlexNet model with a modified classifier for single output.
    """

    def __init__(self):
        """
        Initialize the AlexNet model with a modified classifier layer.

        The final classifier layer is adjusted to output a single value for binary classification.
        """
        super(AlexNet, self).__init__()
        self.model = models.alexnet(pretrained=USE_PRETRAINED)
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the AlexNet model.

        :param x: Input tensor of shape (batch_size, channels, height, width)
        :return: Sigmoid activated output tensor representing probability of the positive class
        """
        return self.sigmoid(self.model(x))


class VGG19(nn.Module):
    """
    VGG19 model for binary classification of images.
    Uses a pretrained VGG19 model with batch normalization and a modified classifier for single output.
    """

    def __init__(self):
        """
        Initialize the VGG19 model with a modified classifier layer.

        The final classifier layer is adjusted to output a single value for binary classification.
        """
        super(VGG19, self).__init__()
        self.model = models.vgg19_bn(pretrained=USE_PRETRAINED)
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the VGG19 model.

        :param x: Input tensor of shape (batch_size, channels, height, width)
        :return: Sigmoid activated output tensor representing probability of the positive class
        """
        return self.sigmoid(self.model(x))
