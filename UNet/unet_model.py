import torch.nn as nn

from UNet.unet_parts import ConvBlock, Decoder, Encoder


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_channels=32) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_channels = base_channels

        self.in_conv = ConvBlock(n_channels, base_channels)
        self.enc_1 = Encoder(base_channels, base_channels * 2)
        self.enc_2 = Encoder(base_channels * 2, base_channels * 4)
        self.enc_3 = Encoder(base_channels * 4, base_channels * 8)
        self.enc_4 = Encoder(base_channels * 8, base_channels * 16)

        self.dec_1 = Decoder(base_channels * 16, base_channels * 8)
        self.dec_2 = Decoder(base_channels * 8, base_channels * 4)
        self.dec_3 = Decoder(base_channels * 4, base_channels * 2)
        self.dec_4 = Decoder(base_channels * 2, base_channels)
        self.out_conv = nn.Conv2d(base_channels, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)

        x2 = self.enc_1(x1)
        x3 = self.enc_2(x2)
        x4 = self.enc_3(x3)
        x5 = self.enc_4(x4)

        x = self.dec_1(x5, x4)
        x = self.dec_2(x, x3)
        x = self.dec_3(x, x2)
        x = self.dec_4(x, x1)

        x = self.out_conv(x)
        return x
