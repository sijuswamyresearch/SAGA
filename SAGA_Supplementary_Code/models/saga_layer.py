import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    """Also known as SiLU (Sigmoid Linear Unit)."""
    def __init__(self): 
        super().__init__()
    def forward(self, x): 
        return x * torch.sigmoid(x)

class FReLU(nn.Module):
    """Spatial baseline: Funnel Activation for Visual Recognition."""
    def __init__(self, channels):
        super().__init__()
        if not isinstance(channels, int) or channels <= 0: 
            raise ValueError(f"FReLU requires positive int channels, got {channels}")
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        if x.dim() != 4: 
            raise ValueError(f"FReLU expects 4D input (NCHW), got {x.dim()}D")
        if x.size(1) != self.channels: 
            raise ValueError(f"Input channels {x.size(1)} != FReLU channels {self.channels}")
        spatial_cond = self.bn(self.conv(x))
        out = torch.max(x, spatial_cond)
        return out

class SAGA(nn.Module):
    """Proposed: Spatially-Adaptive Gated Activation."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # Spatial context extractor
        self.spatial_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.spatial_bn = nn.BatchNorm2d(channels)
        # Dynamic gate generator
        self.gate_generator = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)
        
        # Initialization
        nn.init.kaiming_normal_(self.spatial_conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.spatial_bn.weight, 1)
        nn.init.constant_(self.spatial_bn.bias, 0)
        nn.init.constant_(self.gate_generator.weight, 0)
        nn.init.constant_(self.gate_generator.bias, 2.0) # Starts gate near ~0.88 for stability

    def forward(self, x):
        T_x = self.spatial_bn(self.spatial_conv(x))
        boost = F.relu(T_x - x)
        gate = torch.sigmoid(self.gate_generator(T_x))
        return x + (gate * boost)


def get_activation_instance(act_template, channels_for_act):
    """
    Helper mapping to instantiate the correct activation dynamically 
    during network construction, injecting channel dimensions where needed.
    """
    # Handle custom parametric activations that require channel dimensions
    if isinstance(act_template, FReLU): 
        return FReLU(channels=channels_for_act)
    elif isinstance(act_template, SAGA): 
        return SAGA(channels=channels_for_act)
    
    # Handle standard parameterless activations (Swish, ReLU, ELU, Tanh, etc.)
    else: 
        return copy.deepcopy(act_template)