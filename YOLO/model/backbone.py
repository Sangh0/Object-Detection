from typing import *

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Convolutional Layer, Conv + BN + Leaky ReLU

    Args:
        in_dim: The number of input channels
        out_dim: The number of next channels
        kernel_size: The kernel size for nn.Conv2d
        stride: The stride for nn.Conv2d
        padding: The size of zero-padding for nn.Conv2d
        slope: The slope of nn.LeakyReLU
    """
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        kernel_size: int=3, 
        stride: int=1, 
        padding: int=1,
        slope: float=0.1,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.leaky_relu = nn.LeakyReLU(slope)
    
    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual Block to build Darknet backbone

    Args:
        dim: The number of channels
    """
    def __init__(self, dim: int, hidden_dim: Optional[int]=None):
        super(ResidualBlock, self).__init__()
        assert dim % 2 == 0, f'{in_dim} is not even'
        hidden_dim = dim // 2 if hidden_dim is None else hidden_dim
        self.conv1 = ConvBlock(dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBlock(hidden_dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return x


class DarkNetBackbone(nn.Module):
    """
    Build backbone network to extract features

    Args:
        in_dim: The number of channels
        num_filters: The number of next channels
        repeat_list: repeat count list to build DarkNet53
        task: select between detection and classification, different output
    """
    def __init__(
        self,
        in_dim: int=3, 
        num_filters: int=32, 
        repeat_list: List[int]=[1,2,8,8,4],
        task: str='detection',
    ):
        super(DarkNetBackbone, self).__init__()
        assert len(repeat_list) == 5
        assert task in ('classification', 'detection')
        self.task = task
        
        self.conv = ConvBlock(in_dim, num_filters, kernel_size=3, stride=1, padding=1)
        self.block1 = self._make_darknet_block(num_filters, num_filters*2, res_repeat=repeat_list[0])
        self.block2 = self._make_darknet_block(num_filters*2, num_filters*4, res_repeat=repeat_list[1])
        self.block3 = self._make_darknet_block(num_filters*4, num_filters*8, res_repeat=repeat_list[2])
        self.block4 = self._make_darknet_block(num_filters*8, num_filters*16, res_repeat=repeat_list[3])
        self.block5 = self._make_darknet_block(num_filters*16, num_filters*32, res_repeat=repeat_list[4])

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.block1(x)
        x = self.block2(x)
        out3 = self.block3(x) # 256 x 52 x 52
        out2 = self.block4(out3) # 512 x 26 x 26
        out1 = self.block5(out2) # 1024 x 13 x 13

        if self.task == 'detection':
            return out1, out2, out3
        else:
            return out1

    def _make_darknet_block(self, in_dim, out_dim, res_repeat):
        layers = []
        layers.append(ConvBlock(in_dim, out_dim, kernel_size=3, stride=2, padding=1))
        for _ in range(res_repeat):
            layers.append(ResidualBlock(out_dim))
        return nn.Sequential(*layers)


class DarkNet53(nn.Module):
    """
    Build a model for classification task
    reference: https://pjreddie.com/darknet/imagenet/
    """
    def __init__(
        self,
        in_dim: int=3,
        num_filters: int=32,
        repeat_list: List[int]=[1,2,8,8,4],
        num_classes: int=1000,
        task: str='classification',
    ):
        super(DarkNet53, self).__init__()

        self.features = DarkNetBackbone(
            in_dim=in_dim, 
            num_filters=num_filters, 
            repeat_list=repeat_list,
            task='classification',
        )

        out_features = 1024
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten(start_dim=1),
            nn.Linear(out_features, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.classifier(x)
        return x