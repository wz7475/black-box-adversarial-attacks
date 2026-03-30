import torch
import torch.nn as nn
import torchvision.transforms as transforms


# MNIST CNN
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 64, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

# CIFAR-10 CNN
class CIFARModel(nn.Module):
    def __init__(self):
        super(CIFARModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(8 * 8 * 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# HuggingFace ResNet wrapper (e.g. microsoft/resnet-18)
class HFResNetModel(nn.Module):
    """Wraps a HuggingFace AutoModelForImageClassification.

    Accepts images as float tensors in [0, 1] and applies standard ImageNet
    normalisation internally, so the attack code never needs to know about it.
    After loading, ``model.id2label`` exposes the {int -> class-name} mapping.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, model_name: str = "microsoft/resnet-18"):
        super().__init__()
        from transformers import AutoModelForImageClassification
        self._hf_model = AutoModelForImageClassification.from_pretrained(model_name)
        self._normalize = transforms.Normalize(
            mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD
        )
        self.id2label: dict = dict(self._hf_model.config.id2label)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] float tensor in [0, 1]. Returns raw logits."""
        return self._hf_model(self._normalize(x)).logits

