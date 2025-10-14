from typing import Tuple
import torch
import torch.nn as nn

def _get_mobilenet_v3_small(pretrained: bool = True):
    try:
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        if pretrained:
            m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        else:
            m = mobilenet_v3_small(weights=None)
    except Exception:
        from torchvision.models import mobilenet_v3_small
        m = mobilenet_v3_small(pretrained=pretrained)
    features = m.features
    last_ch = 576
    return features, last_ch

class MobileNetV3Dual(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, healthy_idx: int = 0):
        super().__init__()
        self.features, last_ch = _get_mobilenet_v3_small(pretrained=pretrained)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(last_ch)
        self.relu = nn.ReLU(inplace=True)
        self.head_bin = nn.Linear(last_ch, 2)
        self.head_cls = nn.Linear(last_ch, num_classes)
        self.healthy_idx = healthy_idx

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.bn(x)
        x = self.relu(x)
        return self.head_bin(x), self.head_cls(x)
