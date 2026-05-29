import torch
import torch.nn as nn

class BaseDefense(nn.Module):
    """
    Abstract base class for all pre-processing defenses.
    Defenses act as PyTorch modules that pre-process adversarial inputs
    before feeding them to the target classifier.
    """
    def __init__(self):
        super(BaseDefense, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the forward pass representing the defense pre-processing.")
