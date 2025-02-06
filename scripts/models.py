import torch.nn as nn
from torchvision import models
import torch

# Flag to determine whether to use pretrained weights for the models
USE_PRETRAINED = True


class MobileNetV2(nn.Module):
    """
    MobileNetV2 model for binary classification of images.

    """
    def __init__(self):
        """
        Initialize the MobileNetV2 model with a modified classifier layer.

        """
        super(MobileNetV2, self).__init__()
        # Load the pretrained MobileNetV2 model from torchvision
        self.model = models.mobilenet_v2(pretrained=USE_PRETRAINED)
        # Modify the final classifier layer to output a single value (for binary classification)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 1)
        # Apply sigmoid activation to the output to map the logits to a probability (0 or 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """


        :param x:
        :return:
        """
        # Pass the input through the model and apply the sigmoid activation function
        return self.sigmoid(self.model(x))


class ResNet(nn.Module):
    """
    ResNet model for binary classification of images.

    """
    def __init__(self):
        """
        Initialize the ResNet model with a modified fully connected layer.

        """
        super(ResNet, self).__init__()
        # Load the pretrained ResNet50 model from torchvision
        self.model = models.resnet50(pretrained=USE_PRETRAINED)
        # Modify the fully connected layer to output a single value (for binary classification)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        # Apply sigmoid activation to the output to map the logits to a probability (0 or 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """


        :param x:
        :return:
        """
        # Pass the input through the model and apply the sigmoid activation function
        return self.sigmoid(self.model(x))


class AlexNet(nn.Module):
    """
    AlexNet model for binary classification of images.

    """
    def __init__(self):
        """
        Initialize the AlexNet model with a modified classifier layer.

        """
        super(AlexNet, self).__init__()
        # Load the pretrained AlexNet model from torchvision
        self.model = models.alexnet(pretrained=USE_PRETRAINED)
        # Modify the final classifier layer to output a single value (for binary classification)
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, 1)
        # Apply sigmoid activation to the output to map the logits to a probability (0 or 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """


        :param x:
        :return:
        """
        # Pass the input through the model and apply the sigmoid activation function
        return self.sigmoid(self.model(x))


class VGG19(nn.Module):
    """
    VGG19 model for binary classification of images.

    """
    def __init__(self):
        """
        Initialize the VGG19 model with a modified classifier layer.

        """
        super(VGG19, self).__init__()
        # Load the pretrained VGG19 model with batch normalization from torchvision
        self.model = models.vgg19_bn(pretrained=USE_PRETRAINED)
        # Modify the final classifier layer to output a single value (for binary classification)
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, 1)
        # Apply sigmoid activation to the output to map the logits to a probability (0 or 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """


        :param x:
        :return:
        """
        # Pass the input through the model and apply the sigmoid activation function
        return self.sigmoid(self.model(x))
