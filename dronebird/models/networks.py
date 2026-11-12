"""The two networks, as nn.Module subclasses.

Author: Mariah Habak

SmallCNN: baseline 2, trained from scratch (three conv blocks, as in
the labs). ResNet18Net: torchvision ResNet-18 with ImageNet weights
(declared pretrained model, https://pytorch.org/vision/stable/models/resnet.html)
and a new 2-class head. Both expose ``cam_layer`` for Grad-CAM.
"""

from torch import nn


class SmallCNN(nn.Module):
    """(N, 3, 96, 96) -> (N, 2)."""

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1),
                                 nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                                 nn.Conv2d(c_out, c_out, 3, padding=1),
                                 nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(128, 2))

    def forward(self, x):
        return self.head(self.features(x))

    def cam_layer(self) -> nn.Module:
        return self.features[-1]  # last block, 12 x 12 maps for a 96 px input


class ResNet18Net(nn.Module):
    """(N, 3, 96, 96) -> (N, 2)."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 2))

    def forward(self, x):
        return self.backbone(x)

    def cam_layer(self) -> nn.Module:
        return self.backbone.layer3  # 6 x 6 maps for a 96 px input


def build_network(kind: str, **settings) -> nn.Module:
    if kind == "small_cnn":
        return SmallCNN(dropout=settings.get("dropout", 0.3))
    if kind == "resnet18":
        return ResNet18Net(pretrained=settings.get("pretrained", True),
                           dropout=settings.get("dropout", 0.3))
    raise ValueError(f"Unknown network '{kind}'")
