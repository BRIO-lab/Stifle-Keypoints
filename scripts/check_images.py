"""
Sasank Desaraju
3/14/23

This script is to plot the ground truth keypoints on all the images to make sure they are correct.
"""


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io


os.chdir('/home/sasank/Documents/GitRepos/Stifle-Keypoints/')

#from config.debug import Configuration
from config.debug import Configuration
from KeypointDataset import KeypointDataset

# Import the config file
config = Configuration()

# Create a dataset object with the config file as an argument
dataset = KeypointDataset(config, 'train')

# Get each batch and print the image with the keypoints on top of it

while dataset.index < len(dataset):
    batch = dataset.get_batch()
    image = batch['image']
    keypoints = batch['keypoints']

    # Plot the image with the keypoints on top of it
    plt.imshow(image)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c='r')
    plt.show()