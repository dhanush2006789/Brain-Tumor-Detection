import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SegNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Decoder
        self.dec4 = self.conv_block(512, 256)
        self.dec3 = self.conv_block(256, 128)
        self.dec2 = self.conv_block(128, 64)
        self.dec1 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoding
        x1 = self.enc1(x)
        x1p, idx1 = self.pool(x1)

        x2 = self.enc2(x1p)
        x2p, idx2 = self.pool(x2)

        x3 = self.enc3(x2p)
        x3p, idx3 = self.pool(x3)

        x4 = self.enc4(x3p)
        x4p, idx4 = self.pool(x4)

        # Decoding
        x4u = self.unpool(x4p, idx4)
        x4d = self.dec4(x4u)

        x3u = self.unpool(x4d, idx3)
        x3d = self.dec3(x3u)

        x2u = self.unpool(x3d, idx2)
        x2d = self.dec2(x2u)

        x1u = self.unpool(x2d, idx1)
        out = self.dec1(x1u)

        return out
