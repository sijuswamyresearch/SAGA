import torch
import torch.nn as nn
from .saga_layer import get_activation_instance

class ResidualBlock_DRN(nn.Module):
    """A standard residual block without scaling."""
    def __init__(self, n_feats, kernel_size, activation_fn_template):
        super(ResidualBlock_DRN, self).__init__()
        current_activation = get_activation_instance(activation_fn_template, n_feats)
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2)),
            current_activation,
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2))
        )
        
    def forward(self, x):
        res = self.body(x)
        res += x 
        return res

class DeblurResNet(nn.Module):
    """A ResNet-style architecture for deblurring with a global skip connection."""
    def __init__(self, n_channels_in=1, n_channels_out=1, n_resblocks=16, n_feats=64, activation_fn_template=nn.ReLU()):
        super(DeblurResNet, self).__init__()
        kernel_size = 3
        self.model_n_channels_in = n_channels_in

        # Head and Body
        self.head = nn.Conv2d(n_channels_in, n_feats, kernel_size, padding=(kernel_size//2))
        m_body = [ResidualBlock_DRN(n_feats, kernel_size, activation_fn_template) for _ in range(n_resblocks)]
        self.body = nn.Sequential(*m_body)
        
        # Tail
        self.tail = nn.Conv2d(n_feats, n_channels_out, kernel_size, padding=(kernel_size//2))
        
        # Robust Global Skip Connection handling
        if n_channels_in != n_channels_out:
            self.global_skip = nn.Conv2d(n_channels_in, n_channels_out, kernel_size=1)
        else:
            self.global_skip = nn.Identity()

        self.final_activation = nn.Tanh()

    def forward(self, x):
        if x.dim() != 4 or x.shape[1] != self.model_n_channels_in:
             raise ValueError(f"DeblurResNet Input shape mismatch. Expected (N, {self.model_n_channels_in}, H, W), got {x.shape}")
        
        x_skip = self.global_skip(x) # Ensures dimensions match 'reconstructed' perfectly
        
        features = self.head(x)
        features = self.body(features)
        reconstructed = self.tail(features)
        
        reconstructed += x_skip # Global skip connection with guaranteed dimension match 

        if hasattr(self, 'final_activation') and self.final_activation is not None:
            x_out = self.final_activation(reconstructed)
        else:
            x_out = reconstructed
            
        return x_out