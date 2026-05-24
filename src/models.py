"""Model factory for baseline and transfer-learning classifiers."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights

from src.baseline_model import BaselineCNN

MODEL_CHOICES = ("baseline", "resnet18", "efficientnet_b0")


def create_model(model_name: str, num_classes: int = 2) -> nn.Module:
    """Build a classifier by name."""
    if model_name == "baseline":
        return BaselineCNN(num_classes=num_classes)
    if model_name == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unknown model: {model_name}. Choose from {MODEL_CHOICES}.")


def uses_pretrained_backbone(model_name: str) -> bool:
    return model_name in {"resnet18", "efficientnet_b0"}


def build_optimizer(
    model: nn.Module,
    model_name: str,
    lr: float,
    backbone_lr: float | None = None,
) -> torch.optim.Optimizer:
    """Use a lower learning rate for pretrained backbones and a higher rate for the head."""
    if model_name == "baseline":
        return torch.optim.Adam(model.parameters(), lr=lr)

    backbone_lr = backbone_lr if backbone_lr is not None else lr * 0.1
    backbone_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []

    if model_name == "resnet18":
        head_prefix = "fc."
    elif model_name == "efficientnet_b0":
        head_prefix = "classifier."
    else:
        raise ValueError(f"Unknown model: {model_name}")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(head_prefix):
            head_params.append(param)
        else:
            backbone_params.append(param)

    return torch.optim.Adam(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": lr},
        ]
    )
