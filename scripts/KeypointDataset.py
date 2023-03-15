"""
Sasank Desaraju
3/8/2023

Creating a dataset for keypoint estimation that makes use of the "big_data" CSV files.
"""


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from skimage import io
import cv2

import pytorch_lightning as pl
import wandb

from loss import kp_loss



class KeypointDataset(torch.utils.data.Dataset):

    def __init__(self, config, evaluation_type, transform=None):
        """
        Args:
            config (config): Dictionary of vital constants about data.
            store_data_ram (boolean): Taken from config.
            evaluation_type (string): Dataset evaluation type (must be 'training', 'validation', or 'test')
            num_points (int): Taken from config.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Create local copies of the arguments
        self.config = config
        self.num_points = self.config.dataset['NUM_KEY_POINTS']
        self.transform = self.config.transform
        
        # Check that evaluation_type is valid and then store
        if evaluation_type in ['train', 'val', 'test', 'naive']:
            self.evaluation_type = evaluation_type
        else:
            raise Exception('Incorrect evaluation type! Must be either \'train\', \'val\', \'test\', or \'naive\'.')

        # Load the data from the big_data CSV file into a pandas dataframe
        self.data = pd.read_csv(os.path.join(self.config.etl['DATA_DIR'], self.config.dataset['DATA_NAME'], self.evaluation_type + '_' + self.config.dataset['DATA_NAME'] + '.csv'))

    def __len__(self):
        return len(self.data) - 1   # Subtract 1 because the first row is the column names
    
    def __getitem__(self, idx):
        # Get the row of the dataframe
        row = self.data.iloc[idx]

        # Get the image name
        image_name = row['Image address']

        # Get the image
        image = io.imread(os.path.join(self.config.datamodule['IMAGE_DIRECTORY'], image_name))
        full_image = image             # Save full image (no subset_pixels) for visualization

        # Get the keypoint labels and segmentation labels
        if self.config.dataset['MODEL_TYPE'] == 'fem':
            kp_label = row['Femur 2D KP points']
            seg_label = io.imread(os.path.join(self.config.datamodule['IMAGE_DIRECTORY'], row['Fem label address']))
        elif self.config.dataset['MODEL_TYPE'] == 'tib':
            kp_label = row['Tibia 2D KP points']
            seg_label = io.imread(os.path.join(self.config.datamodule['IMAGE_DIRECTORY'], row['Tib label address']))
        else:
            raise Exception('Incorrect model type! Must be either \'fem\' or \'tib\'.')

        kp_label = kp_label[2:-2]
        kp_label = kp_label.split(']\n [')
        kp_label = [np.array([float(x) for x in list(filter(None, kp.split(' ')))]) for kp in kp_label]
        kp_label = np.array(kp_label)
        
        # * Subset Pixels
        if self.config.dataset['SUBSET_PIXELS'] == True:
            label_dst = np.zeros_like(seg_label)
            label_normed = cv2.normalize(seg_label, label_dst, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX)
            seg_label = label_normed

            kernel = np.ones((30,30), np.uint8)
            label_dilated = cv2.dilate(seg_label, kernel, iterations = 5)
            image_subsetted = cv2.multiply(label_dilated, image)
            image = image_subsetted

        image = torch.FloatTensor(image[None, :, :]) # Store as byte (to save space) then convert when called in __getitem__
        full_image = torch.FloatTensor(full_image[None, :, :]) # Store as byte (to save space) then convert when called in __getitem__
        seg_label = torch.FloatTensor(seg_label[None, :, :])
        #kp_label = torch.FloatTensor(kp_label.reshape(-1))      # Reshape to 1D array so that it's 2*num_keypoints long
        kp_label = torch.FloatTensor(kp_label)          # kp_label is of shape (num_keypoints, 2)
        assert kp_label.shape == (self.num_points, 2), "Keypoint label shape is incorrect!"
        #print("kp_label.shape:")
        #print(kp_label.shape)


    
        # Form sample and transform if necessary
        sample = {'image': image,
                    'img_name': image_name,
                    'kp_label': kp_label,
                    'seg_label': seg_label,
                    'full_image': full_image}
        assert self.transform is None, "Transforms not implemented yet!"
        if self.transform and self.config.dataset['USE_ALBUMENTATIONS'] == True:
            sample = self.transform(sample)     # TODO: Implement transforms. Seems like we'll have to reshape the keypoints to be num_keypoints x 2 instead of 2*num_keypoints x 1
        return sample