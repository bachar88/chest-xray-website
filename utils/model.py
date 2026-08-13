"""
model.py
--------
Defines the neural network architecture.

WHY TRANSFER LEARNING:
Training a CNN from scratch needs millions of images to learn basic things
like "edges", "textures", "shapes". We don't have millions of X-rays.

Instead, we use a model already trained on ImageNet (1.4 million everyday
photos - dogs, cars, furniture, etc). That training taught the model general
visual pattern recognition. We then "fine-tune" it: keep its learned visual
knowledge, but retrain the final layers to recognize OUR classes
(Normal, COVID, Lung_Opacity, Viral_Pneumonia) instead of ImageNet's 1000
classes.

This is standard practice in medical imaging AI because labeled medical
data is always scarce compared to general photos.

We use ResNet50 - a well-established, reliable architecture. DenseNet121
and EfficientNet are also popular in medical imaging; ResNet50 is a great
starting point because it trains fast and is well understood.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes, freeze_backbone=True):
    """
    Loads a pretrained ResNet50 and replaces its final classification layer.

    num_classes: how many output classes (e.g. 4 for Normal/COVID/Opacity/Pneumonia)
    freeze_backbone: if True, we freeze all the pretrained layers and only
                      train the new final layer. This is faster and works
                      well with smaller datasets. You can unfreeze later
                      for a small accuracy boost ("fine-tuning") once the
                      new layer has learned something reasonable.
    """
    # Load ResNet50 pretrained on ImageNet
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        # Freeze all existing layers -- their weights won't update during training
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully-connected layer.
    # ResNet50's original final layer outputs 1000 values (ImageNet classes).
    # We replace it with a small head that outputs `num_classes` values instead.
    in_features = model.fc.in_features  # 2048 for ResNet50
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),          # randomly zero 30% of neurons during training to reduce overfitting
        nn.Linear(256, num_classes)
    )

    return model


def unfreeze_backbone(model, num_layers_from_end=20):
    """
    Optional step 2 of training: unfreeze the last N layers of the backbone
    so they can be fine-tuned slightly to our specific data. Use a much
    smaller learning rate when doing this (see train.py).
    """
    params = list(model.parameters())
    for param in params[-num_layers_from_end:]:
        param.requires_grad = True
    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path, num_classes, device="cpu"):
    model = build_model(num_classes, freeze_backbone=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
