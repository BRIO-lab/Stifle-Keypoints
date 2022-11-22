"""
Sasank Desaraju
11/19/2022

Making custom loss function for Keypoint Estimator.
"""

import torch
import numpy as np
import pandas as pd
import os

class kp_loss(torch.nn.Module):
    def __init__(self, gaussian_amp=1, gaussian_sigma=1):
        super(kp_loss, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.gaussian_amp = gaussian_amp
        self.gaussian_sigma = gaussian_sigma
    
    def gaussian(self, x, y):
        """
        Gaussian function for loss function.
        """
        return self.gaussian_amp * torch.exp(-torch.sum((x - y)**2) / (2 * self.gaussian_sigma**2))
    
    def forward(self, output, target):
        """
        Loss function for Keypoint Estimator.

        output and target are each a batch.
        So, I want output and target to each be a tensor of shape (batch_size, num_keypoints, 2).
        I need to edit the dataset so that the output is the x and y coordinates of the keypoints
        scaled to the image size. Target should also be the same thing, except for these
        the coordinates may have magnitudes that are greater than 1. This will happen in the
        case the case that the keypoints run off the image.
        """
        
        batch_size = target.shape[0]
        raw_batch_loss = 0
        
        for batch_idx, element in enumerate(target):
            for kp_idx, kp in enumerate(element):
                raw_batch_loss += self.gaussian(output[batch_idx][kp_idx], target[batch_idx][kp_idx])
        
        return raw_batch_loss / batch_size





















        # Wow these Copilot print statements are something

        #print("Outputs size: " + str(outputs.size()))
        #print("Labels size: " + str(labels.size()))
        #print("Outputs type: " + str(type(outputs)))
        #print("Labels type: " + str(type(labels)))
        #print("Outputs: " + str(outputs))
        #print("Labels: " + str(labels))
        #print("Outputs shape: " + str(outputs.shape))
        #print("Labels shape: " + str(labels.shape))
        #print("Outputs dtype: " + str(outputs.dtype))
        #print("Labels dtype: " + str(labels.dtype))
        #print("Outputs device: " + str(outputs.device))
        #print("Labels device: " + str(labels.device))
        #print("Outputs is_cuda: " + str(outputs.is_cuda))
        #print("Labels is_cuda: " + str(labels.is_cuda))
        #print("Outputs is_contiguous: " + str(outputs.is_contiguous()))
        #print("Labels is_contiguous: " + str(labels.is_contiguous()))
        #print("Outputs is_pinned: " + str(outputs.is_pinned()))
        #print("Labels is_pinned: " + str(labels.is_pinned()))
        #print("Outputs is_leaf: " + str(outputs.is_leaf))
        #print("Labels is_leaf: " + str(labels.is_leaf))
        #print("Outputs is_shared: " + str(outputs.is_shared()))
        #print("Labels is_shared: " + str(labels.is_shared()))
        #print("Outputs is_set_to_none: " + str(outputs.is_set_to_none()))
        #print("Labels is_set_to_none: " + str(labels.is_set_to_none()))
        #print("Outputs is_sparse: " + str(outputs.is_sparse))
        #print("Labels is_sparse: " + str(labels.is_sparse))
        #print("Outputs is_volatile: " + str(outputs.is_volatile))
        #print("Labels is_volatile: " + str(labels.is_volatile))
        #print("Outputs is_view: " + str(outputs.is_view()))
        #print("Labels is_view: " + str(labels.is_view()))
        #print("Outputs layout: " + str(outputs.layout))
        #print("Labels layout: " + str(labels.layout))
        #print("Outputs numel: " + str(outputs.numel()))