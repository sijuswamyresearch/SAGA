import torch
import torch.nn as nn
from .saga_layer import get_activation_instance

class ResidualBlock_EDSR(nn.Module):
    """
    Standard EDSR residual block with a scaling factor applied 
    to the residual path before addition, improving deep training stability.
    """
    def __init__(self, n_feats, kernel_size, activation_fn_template, res_scale=0.1):
        super(ResidualBlock_EDSR, self).__init__()
        self.res_scale = res_scale
        current_activation = get_activation_instance(activation_fn_template, n_feats)
        
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2)),
            current_activation,
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2))
        )
        
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class EDSR_Deblur(nn.Module):
    """
    Enhanced Deep Residual Network (EDSR) adapted for medical image deblurring.
    Features a deep series of scaled residual blocks and a long skip connection.
    """
    def __init__(self, n_channels_in=1, n_channels_out=1, n_resblocks=16, n_feats=64, activation_fn_template=nn.ReLU(), res_scale=0.1):
        super(EDSR_Deblur, self).__init__()
        kernel_size = 3
        self.model_n_channels_in = n_channels_in
        
        # 1. Head: Projects input channels to feature space
        self.head = nn.Conv2d(n_channels_in, n_feats, kernel_size, padding=(kernel_size//2))
        
        # 2. Body: Deep sequence of scaled residual blocks
        m_body = [
            ResidualBlock_EDSR(n_feats, kernel_size, activation_fn_template, res_scale) 
            for _ in range(n_resblocks)
        ]
        self.body = nn.Sequential(*m_body)
        
        # 3. Body Skip Convolution: Processes features before adding the long skip connection
        self.body_skip_conv = nn.Conv2d(n_feats, n_feats, kernel_size=1) 
        
        # 4. Tail: Reconstructs the image from feature space to output channels
        self.tail = nn.Conv2d(n_feats, n_channels_out, kernel_size, padding=(kernel_size//2))
        
        # 5. Output Normalization
        self.final_activation = nn.Tanh() 

    def forward(self, x):
        if x.dim() != 4 or x.shape[1] != self.model_n_channels_in:
             raise ValueError(f"EDSR Input shape mismatch. Expected (N, {self.model_n_channels_in}, H, W), got {x.shape}")
             
        # Extract shallow features
        x_head = self.head(x)
        
        # Deep feature extraction
        res_body = self.body(x_head)
        res_body_processed = self.body_skip_conv(res_body) 
        
        # EDSR long skip connection (adds shallow features to deep features)
        res_body_processed += x_head 
        
        # Image reconstruction
        x_out = self.tail(res_body_processed)
        
        # Final activation to ensure output stays within [-1, 1] bounds
        if hasattr(self, 'final_activation') and self.final_activation is not None:
            x_out = self.final_activation(x_out)
            
        return x_out