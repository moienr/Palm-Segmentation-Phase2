"""
Generator model for CycleGAN

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-05: Initial coding
* 2022-12-21: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels,out_channels, num_features=64, num_residuals=9):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        
        # This is for the identity loss, so that the model can learn to copy the input image
        # since the input and output may have different number of channels, we have a backup initial layer
        # that does a convolution to match the number of channels
        self.initial_identity = nn.Sequential(
            nn.Conv2d(
                out_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features, num_features * 2, kernel_size=3, stride=2, padding=1
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features * 4,
                    num_features * 2,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 1,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )

        self.last = nn.Conv2d(
            num_features * 1,
            out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )
        

    def forward(self, x):
        """
        Input:
            x: (B, in_channels, H, W) tensor
        Output:
            out: (B, out_channels, H, W), (B, in_channels, H, W) | the second output is for the identity loss.
        
        """
        if x.shape[1] == self.in_channels:
            x = self.initial(x)
        elif x.shape[1] == self.out_channels:
            x = self.initial_identity(x)
        else:
            raise Exception("Input and output channels do not match!")
        
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))


def test():
    in_channels = 1
    out_channels = 3
    img_size = 256
    x = torch.randn((5, in_channels, img_size, img_size))
    gen = Generator(in_channels, out_channels)
    gen_fake = gen(x)
    print(f"gen_fake: {gen_fake.shape}")
    
    # test identity
    identity = gen(gen_fake)
    print(f"identity: {identity.shape}")
    


if __name__ == "__main__":
    test()
