
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # Initialize weights
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# class UNet(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels):
#         super(UNet, self).__init__()
#
#         # Encoder part with customizable channels
#         self.enc1 = self.conv_block(in_channels, mid_channels[0])
#         self.enc2 = self.conv_block(mid_channels[0], mid_channels[1])
#         self.enc3 = self.conv_block(mid_channels[1], mid_channels[2])
#         self.enc4 = self.conv_block(mid_channels[2], mid_channels[3])
#
#         # Bottleneck part with customizable channels
#         self.bottleneck = self.conv_block(mid_channels[3], mid_channels[4])
#
#         # Decoder part with customizable channels
#         self.dec4 = self.conv_block(mid_channels[4] + mid_channels[3], mid_channels[3])
#         self.dec3 = self.conv_block(mid_channels[3] + mid_channels[2], mid_channels[2])
#         self.dec2 = self.conv_block(mid_channels[2] + mid_channels[1], mid_channels[1])
#         self.dec1 = self.conv_block(mid_channels[1] + mid_channels[0], mid_channels[0])
#
#         # Output layer
#         self.out_conv = nn.Conv2d(mid_channels[0], out_channels, kernel_size=1)
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.sig = nn.Sigmoid()
#
#         # Initialize weights
#         self._initialize_weights()
#
#     def conv_block(self, in_channels, out_channels):
#         block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#         return block
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.ConvTranspose2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         # Encoder part
#         enc1 = self.enc1(x)
#         enc2 = self.enc2(self.maxpool(enc1))
#         enc3 = self.enc3(self.maxpool(enc2))
#         enc4 = self.enc4(self.maxpool(enc3))
#
#         # Bottleneck part
#         bottleneck = self.bottleneck(self.maxpool(enc4))
#
#         # Decoder part
#         dec4 = self.up_concat(bottleneck, enc4)
#         dec4 = self.dec4(dec4)
#
#         dec3 = self.up_concat(dec4, enc3)
#         dec3 = self.dec3(dec3)
#
#         dec2 = self.up_concat(dec3, enc2)
#         dec2 = self.dec2(dec2)
#
#         dec1 = self.up_concat(dec2, enc1)
#         dec1 = self.dec1(dec1)
#
#         # Output layer
#         out = self.out_conv(dec1)
#         out = self.sig(out)
#         return out
#
#     def up_concat(self, x1, x2):
#         x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
#         return torch.cat([x1, x2], dim=1)

# 示例使用
if __name__ == "__main__":
    from torchsummary import summary
    tensor_rand = torch.rand(1, 3, 512, 512).to('cuda')
    mid_channels = [16, 32, 64, 128, 256]
    # model = UNet(in_channels=3, out_channels=1).to('cuda')
    model = UNet(n_channels=3, n_classes=1).to('cuda')
    y = model(tensor_rand)  # 前向传播
    print(y.shape)  # 输出张量的形状
