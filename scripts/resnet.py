"""
Sasank
12/16/22
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Let's make a resnet.
"""

class ResBlock(nn.Module):
    #def __init__(self, in_channels, out_channels, stride=1, downsample=None):

    expansion = 4

    def __init__(self, in_channels, intermediate_channels, resolution_shrink=1):
        super(ResBlock, self).__init__()

        out_channels = intermediate_channels * self.expansion
        self.downsample = None

        if resolution_shrink != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                # identity channel number and resolution taken care of below
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=resolution_shrink, bias=False),
                nn.BatchNorm2d(out_channels)
            )


        self.relu = nn.ReLU()
        # layer channel number take care of below
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        # layer resolution taken care of below
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=resolution_shrink, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x.clone()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_input_channels, num_classes):
        super(ResNet, self).__init__()

        # Define some fields
        # I guess not?

        # Define the initial layer
        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.latest_output_channels = 64

        # Define the residual layers
        self.layer1 = self._make_layer(block, layers[0], intermediate_channels = 64, resolution_shrink = 1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels = 128, resolution_shrink = 2)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels = 256, resolution_shrink = 2)
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels = 512, resolution_shrink = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_blocks, intermediate_channels, resolution_shrink):

        layers = []

        for i in range(num_blocks):
            if i == 0:
                layers.append(block(self.latest_output_channels, intermediate_channels, resolution_shrink))
                self.latest_output_channels = intermediate_channels * block.expansion
                continue
            layers.append(block(intermediate_channels * block.expansion, intermediate_channels))



        return nn.Sequential(*layers)





def test():
    net = ResNet(ResBlock, [3, 4, 6, 3], 3, 100)
    # I think change the image size
    y = net(torch.randn(2, 3, 224, 224)).to('cuda')
    print(y.size())
    #print(net.shape)


if __name__ == '__main__':
    test()































