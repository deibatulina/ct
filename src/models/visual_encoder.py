import torch
from torch import nn

try:
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError:
    from torchvision.models import resnet18

    ResNet18_Weights = None


def build_resnet18():
    if ResNet18_Weights is None:
        return resnet18(pretrained=True)
    return resnet18(weights=ResNet18_Weights.DEFAULT)


def freeze_module(module):
    for parameter in module.parameters():
        parameter.requires_grad = False


class VisualEncoder(nn.Module):
    def __init__(self, encoder_name="resnet18", frozen=True):
        super().__init__()

        if encoder_name != "resnet18":
            raise ValueError(f"Unsupported visual encoder: {encoder_name}")

        backbone = build_resnet18()
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.frozen = frozen

        if frozen:
            freeze_module(self.backbone)

    def forward(self, images):
        batch_size, num_slices, channels, height, width = images.shape
        flat_images = images.view(batch_size * num_slices, channels, height, width)

        if self.frozen:
            with torch.no_grad():
                features = self.backbone(flat_images)
        else:
            features = self.backbone(flat_images)

        features = features.flatten(start_dim=1)
        return features.view(batch_size, num_slices, -1)
