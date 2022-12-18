import torch
import torch.nn as nn
import albumentations as A
import numpy as np
import time
import os

"""
ResNet-based Keypoint Estimator
"""
class Configuration:
    def __init__(self):
        """
        self.temp = {
            'train_data': '/home/sasank/Documents/GitRepos/Sasank_JTML_seg/data/3_2_22_fem/train_3_2_22_fem.csv',
            'val_data': '/home/sasank/Documents/GitRepos/Sasank_JTML_seg/data/3_2_22_fem/val_3_2_22_fem.csv',
            'test_data': '/home/sasank/Documents/GitRepos/Sasank_JTML_seg/data/3_2_22_fem/test_3_2_22_fem.csv'
        }
        """
        self.init = {
            'PROJECT_NAME': 'Keypoint Estimator Development!',
            'MODEL_NAME': 'Tib_16',
            'RUN_NAME': time.strftime('%Y-%m-%d-%H-%M-%S'),
            'WANDB_RUN_GROUP': 'Local',
            'FAST_DEV_RUN': False,  # Runs inputted batches (True->1) and disables logging and some callbacks
            'MAX_EPOCHS': 10,
            'MAX_STEPS': -1,    # -1 means it will do all steps and be limited by epochs
            'STRATEGY': None    # This is the training strategy. Should be 'ddp' for multi-GPU (like HPG)
        }
        self.etl = {
            'RAW_DATA_FILE': 'Tib_16.csv',
            'DATA_DIR': "data",
            'KEYPOINT_DIRECTORY': "keypoints",
            'KEYPOINT_TXT_FILES': ['tib_KPlabels_16.txt'],
            'VAL_SIZE':  0.2,       # looks sus
            'TEST_SIZE': 0.01,      # I'm not sure these two mean what we think
            #'random_state': np.random.randint(1,50)
            # HHG2TG lol; deterministic to aid reproducibility
            'RANDOM_STATE': 42,

            'CUSTOM_TEST_SET': False,
            'TEST_SET_NAME': '/my/test/set.csv'
        }

        self.dataset = {
            'SUBSET_PIXELS': True,
            'IMAGE_HEIGHT': 1024,
            'IMAGE_WIDTH': 1024,
            'MODEL_TYPE': 'tib',        # how should we do this? not clear this is still best...
            'CLASS_LABELS': {0: 'bone', 1: 'background'},
            'NUM_KEY_POINTS': 16,
            'IMG_CHANNELS': 1,      # Is this differnt from self.module['NUM_IMAGE_CHANNELS']
            'STORE_DATA_RAM': False,
            'IMAGE_THRESHOLD': 0,
            'USE_ALBUMENTATIONS': True,

            'NUM_PRINT_IMG' : 1,
            'KP_PLOT_RAD' : 3,

            #'NUM_POINTS' : 128,

            'GAUSSIAN_STDDEV' : 5,
            'GAUSSIAN_AMP' : 1e3,

            'STORE_DATA_RAM' : False,

            'CROP_IMAGES' : False,
            'CROP_MIN_X' : 0.29,
            'CROP_MAX_X' : 0.84,
            'CROP_MIN_Y' : 0.45,
            'CROP_MAX_Y' : 0.95,
            
            'IMAGES_PER_GRID': 1,
            'per_grid_image_count_height' : 1, 
            'per_grid_image_count_width' : 1
        }

        """
        # segmentation_net_module needs to be below dataset because it uses dataset['IMG_CHANNELS']
        self.keypoint_net_module = {
            'NUM_KEY_POINTS': 128,
            'NUM_IMG_CHANNELS': self.dataset['IMG_CHANNELS']
        }
        """

        self.datamodule = {
            'IMAGE_DIRECTORY': '/media/sasank/LinuxStorage/Dropbox (UFL)/Canine Kinematics Data/TPLO_Ten_Dogs_grids/',
            'CKPT_FILE': None,
            'BATCH_SIZE': 1,
            'SHUFFLE': True,        # Only for training, for test and val this is set in the datamodule script to False
            'NUM_WORKERS': 2,
            'PIN_MEMORY': False,
            'SUBSET_PIXELS': True
        }


        # hyperparameters for training
        self.hparams = {
            'LOAD_FROM_CHECKPOINT': False,
            'learning_rate': 1e-3
        }

        self.transform = None
        """
        self.transform = \
        A.Compose([
        A.RandomGamma(always_apply=False, p = 0.5,gamma_limit=(10,300)),
        A.ShiftScaleRotate(always_apply = False, p = 0.5,shift_limit=(-0.06, 0.06), scale_limit=(-0.1, 0.1), rotate_limit=(-180,180), interpolation=0, border_mode=0, value=(0, 0, 0)),
        A.Blur(always_apply=False, blur_limit=(3, 10), p=0.2),
        A.Flip(always_apply=False, p=0.5),
        A.ElasticTransform(always_apply=False, p=0.85, alpha=0.5, sigma=150, alpha_affine=50.0, interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, approximate=False),
        A.InvertImg(always_apply=False, p=0.5),
        A.CoarseDropout(always_apply = False, p = 0.25, min_holes = 1, max_holes = 100, min_height = 25, max_height=25),
        A.MultiplicativeNoise(always_apply=False, p=0.25, multiplier=(0.1, 2), per_channel=True, elementwise=True)
        ],
        p=0.85)
        """
