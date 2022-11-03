"""
Sasank Desaraju
9/13/22
"""

import torch
import pytorch_lightning as pl
import numpy as np
import os
from skimage import io
import cv2

from FeaturePointDataset import FeaturePointDataset


class KeypointDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.img_dir = self.config.datamodule['IMAGE_DIRECTORY']
        self.train_data = os.getcwd() + '/data/' + self.config.init['MODEL_NAME'] + '/' + 'train_' + self.config.init['MODEL_NAME'] + '.csv'
        self.val_data = os.getcwd() + '/data/' + self.config.init['MODEL_NAME'] + '/' + 'val_' + self.config.init['MODEL_NAME'] + '.csv'
        self.test_data = os.getcwd() + '/data/' + self.config.init['MODEL_NAME'] + '/' + 'test_' + self.config.init['MODEL_NAME'] + '.csv'

        # Data loader parameters
        # TODO: clean this up since we pulled config into this class
        self.batch_size = self.config.datamodule['BATCH_SIZE']
        self.num_workers = self.config.datamodule['NUM_WORKERS']
        self.pin_memory = self.config.datamodule['PIN_MEMORY']
        self.shuffle = self.config.datamodule['SHUFFLE']
        self.train_data_loader_parameters = {'batch_size': self.batch_size,
                                            'num_workers': self.num_workers,
                                            'pin_memory': self.pin_memory,
                                            'shuffle': self.shuffle}
        self.val_data_loader_parameters = {'batch_size': self.batch_size,
                                            'num_workers': self.num_workers,
                                            'pin_memory': self.pin_memory,
                                            'shuffle': False}
        self.test_data_loader_parameters = {'batch_size': self.batch_size,
                                            'num_workers': self.num_workers,
                                            'pin_memory': self.pin_memory,
                                            'shuffle': False}

        #self.log(batch_size=self.batch_size)
        # other constants

        """
        self.train_set = np.genfromtxt(self.train_data, delimiter=',', dtype=str)
        self.val_set = np.genfromtxt(self.val_data, delimiter=',', dtype=str)
        self.test_set = np.genfromtxt(self.test_data, delimiter=',', dtype=str)
        """

        # TODO: check train dataset length and integrity
        # TODO: check val dataset length and integrity

    """
    def prepare_data(self):

        return
    """

    def setup(self, stage):

        self.training_set = FeaturePointDataset(config=self.config,
                                            evaluation_type='train')
        self.validation_set = FeaturePointDataset(config=self.config,
                                            evaluation_type='val')
        self.test_set = FeaturePointDataset(config=self.config,
                                            evaluation_type='test')

        return

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.training_set, **self.train_data_loader_parameters)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.validation_set, **self.val_data_loader_parameters)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, **self.test_data_loader_parameters)
