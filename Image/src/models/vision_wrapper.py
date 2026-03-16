import torch
import torch.nn as nn
from torchvision import models

class VisionModelWrapper:
    def __init__(self, model_name="resnet18", device=None):
        self.device = device if device else ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Initializing {model_name} on {self.device}")
        
        # Load pre-trained model
        if model_name.startswith("cifar10_"):
            # Load from chenyaofo's repo
            self.model = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained=True)
        elif model_name == "resnet18":
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == "resnet34":
            self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif model_name == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == "resnet101":
            self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        elif model_name == "resnet152":
            self.model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Performance counters
        self.forward_count = 0
        self.backward_count = 0
        
        # Loss function for classification
        self.criterion = nn.CrossEntropyLoss()

    def get_loss(self, x, y_true):
        """
        Calculate loss and ensure x tracks gradients.
        """
        self.forward_count += 1
        if x.device != self.device:
            x = x.to(self.device)
        
        if not x.requires_grad:
            x.requires_grad_(True)
            
        y_true = y_true.to(self.device)
        outputs = self.model(x)
        loss = self.criterion(outputs, y_true)
        
        return loss, outputs

    def predict(self, x):
        """
        Return the predicted class index.
        """
        with torch.no_grad():
            outputs = self.model(x.to(self.device))
            _, predicted = torch.max(outputs, 1)
        return predicted

    def zero_grad(self):
        self.model.zero_grad()
