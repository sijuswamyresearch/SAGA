import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .saga_layer import get_activation_instance

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, activation_fn=nn.ReLU()):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        if not isinstance(activation_fn, nn.Module): raise TypeError("activation_fn must be an nn.Module instance")
        self.activation1 = get_activation_instance(activation_fn, mid_channels)
        self.activation2 = get_activation_instance(activation_fn, out_channels)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(mid_channels), self.activation1,
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), self.activation2)
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, activation_fn):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, activation_fn=activation_fn))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, activation_fn, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1) 
            conv_in_channels = skip_channels + out_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels , kernel_size=2, stride=2) 
            conv_in_channels = skip_channels + out_channels 
        self.conv = DoubleConv(conv_in_channels, out_channels, activation_fn=activation_fn)
    def forward(self, x1, x2): 
        x1 = self.up(x1)
        if self.bilinear:
            x1 = self.conv_reduce(x1) 
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, activation_fn=nn.ReLU(), bilinear=True): 
        super(UNet, self).__init__()
        if n_channels <= 0 or n_classes <= 0: raise ValueError("n_channels/n_classes must be positive")
        self.n_channels_unet = n_channels
        self.n_classes = n_classes; self.bilinear = bilinear
        self.activation_fn_template = copy.deepcopy(activation_fn)
        c1, c2, c3, c4 = 64, 128, 256, 512; c5 = 1024 
        
        self.inc = DoubleConv(n_channels, c1, activation_fn=self.activation_fn_template)
        self.down1 = Down(c1, c2, activation_fn=self.activation_fn_template)
        self.down2 = Down(c2, c3, activation_fn=self.activation_fn_template)
        self.down3 = Down(c3, c4, activation_fn=self.activation_fn_template)
        factor = 2 if bilinear else 1
        self.down4 = Down(c4, c5 // factor, activation_fn=self.activation_fn_template) 

        self.up1 = Up(c5 // factor, c4, c4 // factor, self.activation_fn_template, bilinear)
        self.up2 = Up(c4 // factor, c3, c3 // factor, self.activation_fn_template, bilinear)
        self.up3 = Up(c3 // factor, c2, c2 // factor, self.activation_fn_template, bilinear)
        self.up4 = Up(c2 // factor, c1, c1, self.activation_fn_template, bilinear) 
        self.outc = nn.Conv2d(c1, n_classes, kernel_size=1)
        self.final_activation = nn.Tanh() 
    def forward(self, x):
        if x.dim() != 4 or x.shape[1] != self.n_channels_unet:
            raise ValueError(f"UNet Input shape mismatch. Expected (N, {self.n_channels_unet}, H, W), got {x.shape}")
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        logits = self.outc(x); output = self.final_activation(logits)
        return output