"""
Sasank
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Okay, what are we trying to do here.
We want to take to take in an image and output keypoint estimates.
(Let's save the pose estimation network for later.)
We want our outputs to give us the 2D locations of keypoints.
So maybe num_channels = num_keypoints and output nodes are 2 doubles (bc 2D) + 1 bool (to indicate if the keypoint is present on the screen or not)
"""
