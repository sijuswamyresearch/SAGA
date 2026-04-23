import torch
import torch.nn as nn
from .saga_layer import get_activation_instance

class PlainVGGNet(nn.Module):
    """
    An 18-layer 'plain' convolutional neural network for image restoration.
    It evaluates the activation functions within a deep, purely sequential 
    framework, relying entirely on a global residual connection to learn 
    the high-frequency sharpening map.
    """
    def __init__(self, n_channels_in=1, n_channels_out=1, n_layers=18, n_feats=64, activation_fn_template=nn.ReLU()):
        super(PlainVGGNet, self).__init__()
        self.model_n_channels_in = n_channels_in
        kernel_size = 3
        padding = kernel_size // 2

        layers = []
        
        # 1. Head (Input projection)
        layers.append(nn.Conv2d(n_channels_in, n_feats, kernel_size, padding=padding))
        layers.append(get_activation_instance(activation_fn_template, n_feats))

        # 2. Body (16 intermediate layers for an 18-layer network)
        for _ in range(n_layers - 2):
            layers.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=padding))
            layers.append(get_activation_instance(activation_fn_template, n_feats))

        # 3. Tail (Output projection)
        layers.append(nn.Conv2d(n_feats, n_channels_out, kernel_size, padding=padding))
        
        self.body = nn.Sequential(*layers)

        # 4. Robust Global Skip Connection
        # As established in the ResNet script, this protects against channel mismatch
        if n_channels_in != n_channels_out:
            self.global_skip = nn.Conv2d(n_channels_in, n_channels_out, kernel_size=1)
        else:
            self.global_skip = nn.Identity()

        # 5. Output Normalization
        self.final_activation = nn.Tanh()

    def forward(self, x):
        if x.dim() != 4 or x.shape[1] != self.model_n_channels_in:
             raise ValueError(f"PlainVGGNet Input shape mismatch. Expected (N, {self.model_n_channels_in}, H, W), got {x.shape}")
        
        # Preserve input for the global residual connection
        x_skip = self.global_skip(x)
        
        # Forward pass through the deep sequential body
        residual_map = self.body(x)
        
        # Apply the global residual connection (learning the residual)
        reconstructed = residual_map + x_skip

        # Final activation to ensure output stays within [-1, 1] bounds
        if hasattr(self, 'final_activation') and self.final_activation is not None:
            x_out = self.final_activation(reconstructed)
        else:
            x_out = reconstructed
            
        return x_out