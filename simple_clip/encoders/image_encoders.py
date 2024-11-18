# Import necessary PyTorch modules
import torch.nn as nn  # Neural network layers and functions
from torchvision import models  # Pre-trained computer vision models
# Import specific pre-trained model weights
from torchvision.models import ResNet50_Weights, ResNet18_Weights, EfficientNet_V2_S_Weights


class Squeeze(nn.Module):
    """
    A custom layer that removes extra dimensions from the output
    This is useful when converting 2D image features to 1D vectors
    """

    def forward(self, x):
        # Remove the last two dimensions (usually 1x1 after average pooling)
        # For example, changes shape from (batch_size, channels, 1, 1) to (batch_size, channels)
        return x.squeeze(-1).squeeze(-1)


def _prep_encoder(model):
    """
    Prepares a pre-trained model to be used as an encoder by:
    1. Removing the classification head (last layer)
    2. Adding adaptive average pooling to get fixed size output
    3. Adding the squeeze layer to flatten the output

    Args:
        model: A pre-trained neural network (like ResNet or EfficientNet)

    Returns:
        A modified version of the model suitable for encoding images into vectors
    """
    # Get all layers except the last one (classification layer)
    modules = list(model.children())[:-1]

    # Add adaptive average pooling layer that outputs 1x1 spatial dimensions
    modules.append(nn.AdaptiveAvgPool2d(1))

    # Add squeeze layer to remove the 1x1 dimensions
    modules.append(Squeeze())

    # Combine all modules into a sequential model
    return nn.Sequential(*modules)


def resnet18():
    """
    Creates a ResNet18 encoder:
    - Uses pre-trained weights from ImageNet
    - Modifies the architecture for encoding

    Returns:
        Modified ResNet18 model for image encoding
    """
    # Load pre-trained ResNet18 with ImageNet weights
    resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Modify it for encoding
    return _prep_encoder(resnet)


def resnet50():
    """
    Creates a ResNet50 encoder:
    - Uses pre-trained weights from ImageNet
    - Modifies the architecture for encoding

    Returns:
        Modified ResNet50 model for image encoding
    """
    # Load pre-trained ResNet50 with ImageNet weights
    resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # Modify it for encoding
    return _prep_encoder(resnet)


def efficientnet_v2_s():
    """
    Creates an EfficientNetV2-S encoder:
    - Uses pre-trained weights from ImageNet
    - Modifies the architecture for encoding

    Returns:
        Modified EfficientNetV2-S model for image encoding
    """
    # Load pre-trained EfficientNetV2-S with ImageNet weights
    model = models.efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    # Modify it for encoding
    return _prep_encoder(model)
