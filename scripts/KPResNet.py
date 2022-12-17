"""
Sasank
12/17/22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class KPResNet(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super(KPResNet, self).__init__()

        self.in_channels = in_channels
        self.num_keypoints = num_keypoints

        self.my_dict = nn.ModuleDict({})

        self.my_dict["pre_block"] = nn.Sequential(
            nn.Conv2d(self.in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        
        #self.my_dict["resnet"] = make_resnet(3, 64, 3, 1)
        # this 
        self.my_dict["resnet"] = torchvision.models.resnet34(pretrained=True)

        self.my_dict["keypoints"] = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 2 * self.num_keypoints)
        )

    def forward(self, x):
        x = self.my_dict["pre_block"](x)
        x = self.my_dict["resnet"](x)
        x = self.my_dict["keypoints"](x)
        return x

def test():
    model = KPResNet(1, 16)
    x = torch.randn(2, 1, 1024, 1024)
    y = model(x)
    print(y.shape)

if __name__ == "__main__":
    test()