import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels,out_channels,kernel_size=3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
            )
    def forward(self,x):
        output = self.double_conv(x)
        return output


class Unet(nn.Module):
    def __init__(self,in_channels = 3,out_channels = 1):
        super(Unet,self).__init__()
        self.enc1 = DoubleConv(inc_channels = in_channels ,out_channels =64)
        self.enc2 = DoubleConv(in_channels = 64,out_channels = 128)
        self.enc3 = DoubleConv(in_channels =128,out_channels =256)
        self.enc4 = DoubleConv(in_channels = 256,out_channels = 512)

        self.bottle_neck = DoubleConv(in_channels = 512, out_channels= 1024)

        self.pool = nn.MaxPool2d(kernel_size=2)


      
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(1024, 512)  # concat with skip connection

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(512, 256)  # concat with skip connection

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)  # concat with skip connection

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(128, 64)  # concat with skip connection

        # Final output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)  # For binary segmentation (out_channels = 1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        # Bottleneck
        x = self.bottleneck(self.pool(x4))

        # Decoder
        x = self.up1(x)
        x = torch.cat([x, x4], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec4(x)

        # Output
        return self.out_conv(x)