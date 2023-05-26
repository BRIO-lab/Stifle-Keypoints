"""
Sasank Desaraju
11/19/2022

Making custom loss function for Keypoint Estimator.
"""

import torch
import numpy as np
import pandas as pd
import os
import math

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
    
    def gaussian_heatmap(self, keypoint, img_h, img_w):
        '''
        Takes a keypoint and returns a heatmap of the keypoint.
        '''
        kp = keypoint

        #heatmap = np.zeros((img_h, img_w))
        # Changing these from np.linspace to torch.linspace and np.meshgrid to torch.meshgrid
        g_x = torch.linspace(0.5, img_w - 0.5, img_w, device=kp.device)
        g_y = torch.linspace(0.5, img_h - 0.5, img_h, device=kp.device)
        g_x, g_y = torch.meshgrid(g_x, g_y)
        g_z = self.gaussian_amp/(2*np.pi*self.gaussian_sigma**2) *torch.exp(
            #-(((g_x - math.floor(point[0]*img_w)+0.5)**2 + (g_y - math.floor(
            #    img_h*(1 - point[1]))+0.5)**2) /(2*GAUSSIAN_STDEV_HEATMAP**2))) # ADDED FLOOR TO MAKE SURE GAUSSIAN ALWAYS PEAKS ON PIXEL
            # Why does the g_y term below have (1 - point[1])? Shouldn't it be (point[1])?
            -(((g_x - (kp[0]*img_w))**2 + (g_y - (img_h*(1 - kp[1])))**2) /(2*self.gaussian_sigma**2)))
        return g_z

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
                #raw_batch_loss += self.gaussian(output[batch_idx][kp_idx], target[batch_idx][kp_idx])
                
                # These asserts are just to make sure the shapes are correct. Can change if we're not doing 1024x1024.
                assert output.shape[-2] == 1024, 'output[-2] is of shape ' + str(output.shape[-2])
                assert output.shape[-1] == 1024, 'output[-1] is of shape ' + str(output.shape[-1])
                target_heatmap = self.gaussian_heatmap(target[batch_idx][kp_idx],
                                                        img_h=output.shape[-2],
                                                        img_w=output.shape[-1])
                raw_batch_loss += self.mse(output[batch_idx][kp_idx], target_heatmap)
        
        return raw_batch_loss / batch_size



class res_kp_loss(torch.nn.Module):
    # TODO: This does not use any Gaussian function. It just uses MSE. Should we remove all the Gaussian stuff?
    def __init__(self, gaussian_amp=1, gaussian_sigma=1):
        super(res_kp_loss, self).__init__()
        #self.mse = torch.nn.MSELoss()
        #self.gaussian_amp = gaussian_amp
        #self.gaussian_sigma = gaussian_sigma

        self.HEIGHT = 1024
        self.WIDTH = 1024
        self.outer_bound_weight = 0.01

    def forward(self, output, target):
        """
        The output is [batch_size, 2 * num_keypoints].
        I'll interpret the i'th prediction as (output[batch_idx][2*i], output[batch_idx][2*i + 1]).
        The target is [batch_size, num_keypoints, 2].
        """

        batch_size = target.shape[0]
        #num_keypoints = int(target.shape[1] / 2)     # Since the output is 2 * num_keypoints per batch.
        num_keypoints = target.shape[1]     # Since the output is 2 * num_keypoints per batch.
        raw_batch_loss = 0

        # The model output is (num_keypoint,2) since we changed the module forward method on 3/16/23
        assert output.shape[1] == num_keypoints, 'output.shape[1] is ' + str(output.shape[1]) + ' but num_keypoints is ' + str(num_keypoints)
        assert output.shape[2] == 2, 'output.shape[2] is ' + str(output.shape[2]) + ' but should be 2'

        for batch_idx, _ in enumerate(target):
            for i in range(0, num_keypoints):
                #raw_batch_loss += self.gaussian(output[batch_idx][i], target[batch_idx][i])

                # We want to de-weight the keypoints which are beyond the image size.
                # Weight is 1 only if the keypoint is within 0 and the image bounds. Negative values are outside the image.
                #weight = 1 if (target[batch_idx][i][0].item() in range(0, self.WIDTH) and target[batch_idx][i][1].item() in range(0, self.HEIGHT)) else self.outer_bound_weight
                weight = 1
                if target[batch_idx][i][0].item() < 0 or target[batch_idx][i][0].item() > self.WIDTH:
                    weight = self.outer_bound_weight 
                if target[batch_idx][i][1].item() < 0 or target[batch_idx][i][1].item() > self.HEIGHT:
                    weight = self.outer_bound_weight
                #weight = 1 if (target[batch_idx][i][0] < self.WIDTH and target[batch_idx][i][1] < self.HEIGHT) else self.outer_bound_weight
                raw_single_kp_loss = self.mse(output[batch_idx][i], target[batch_idx][i])
                weighted_single_kp_loss = weight * raw_single_kp_loss
                raw_batch_loss += weighted_single_kp_loss

                # print the whether the keypoint is within the image bounds and its loss
                #print("kp " + str(i) + " is within bounds: " + str(weight == 1) + " and has loss: " + str(weighted_single_kp_loss))
                # print the target kp
                #print("target kp: " + str(target[batch_idx][i][0].item()) + ", " + str(target[batch_idx][i][1].item()))

        avg_loss = raw_batch_loss / batch_size
        return avg_loss




    
    # ? Should we use the built-in MSE loss function here?
    def mse(self, pred, target):
        #print("pred: " + str(pred))
        #print("target: " + str(target))
        mse_loss = (pred[0] - target[0])**2 + (pred[1] - target[1])**2
        #print("mse_loss: " + str(mse_loss))
        return mse_loss

    def gaussian(self, pred, target):
        """
        pred and target are each a tensor of shape (2).
        The first element of pred and target is the x coordinate
        and the second element is the y coordinate.
        """
        #print("pred shape: " + str(pred.shape))
        #print("target shape: " + str(target.shape))
        #print("pred: " + str(pred))
        #print("pred x: " + str(pred[0]))
        #print("pred datatype: " + str(pred.dtype))
        #print("target: " + str(target))
        #print("target x: " + str(target[0]))
        #print("sum x: " + str(pred[0] + target[0]))
        #print("pred and target: " + str(pred) + " " + str(target))
        #print("pred and target: " + " ".join([str(pred)[0], str(pred)[1], str(target)[0], str(target)[1]]))
        #print(str(pred[0].item()))
        gauss_loss = self.gaussian_amp/(math.sqrt(2*np.pi)*self.gaussian_sigma) *torch.exp(-((pred[0] - target[0])**2 + (pred[1] - target[1])**2) / (2*self.gaussian_sigma**2))
        print("gauss_loss: " + str(gauss_loss))
        return gauss_loss
















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