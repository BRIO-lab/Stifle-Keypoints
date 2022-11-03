########## PUT STUFF IN HERE LIKE HELPER METHODS THAT DONT BELONG IN etl, train, or test 
# (i.e. create_gaussian_heatmap(point_list, img_h, img_w) )
import numpy as np
# from config.default_config import Configuration
import random
import matplotlib.pyplot as plt
import cv2
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from skimage import io, transform
import time
import sys
import glob
import os
import math
import random
from pose_hrnet import PoseHighResolutionNet
import numpy as np
import itertools
import logging
from pathlib import Path



# config = Configuration()
# THIS IS HOW WE SHOULD BE IMPORTING CONFIG, NOT WITH from _____config import ...
#     ##################
#     # Load config from config file
#     ##################
    
#   #  config = parse_config(config_file)

#     # get string for config directory
#     config_dir = os.getcwd() + "/config/"

#     # add config directory to system path so config can be imported as module
#     sys.path.append(config_dir)

#     # import config to use Configuration class
#     config_module = import_module(config_file)

#     # get instance of configuration class
#     config = config_module.Configuration()


def set_logger(log_path):
    """
    Read more about logging: https://www.machinelearningplus.com/python/python-logging-guide/
    Args:
        log_path [str]: eg: "../log/train.log"
    """

    # parameter log_path specifies directory to use for logging
    # should be in the form ./log/[MODEL_NAME]/etl_[MODEL_NAME].log
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # configure the logger object
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, mode="a")
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Finished logger configuration!")
    return logger


def create_gaussian_heatmap(config, point_list, img_h, img_w):
        '''
        Takes list of (x,y) points and returns a list of gaussian heat maps centered at the respective points.
        '''
        gaussian_amp = config.data_constants['GAUSSIAN_AMP']
        gaussian_sdv_heatmap = config.data_constants['GAUSSIAN_STDEV_HEATMAP']


        heat_maps = []
        for point in point_list:
            g_x = np.linspace(0.5, img_w - 0.5, img_w)
            g_y = np.linspace(0.5, img_h - 0.5, img_h)
            g_x, g_y = np.meshgrid(g_x, g_y)
            g_z = gaussian_amp/(2*np.pi*gaussian_sdv_heatmap**2) *np.exp(
                #-(((g_x - math.floor(point[0]*img_w)+0.5)**2 + (g_y - math.floor(
                #    img_h*(1 - point[1]))+0.5)**2) /(2*GAUSSIAN_STDEV_HEATMAP**2))) # ADDED FLOOR TO MAKE SURE GAUSSIAN ALWAYS PEAKS ON PIXEL
                -(((g_x - point[0]*img_w)**2 + (g_y - img_h*(1 - point[1]))**2) /(2*gaussian_sdv_heatmap**2)))
            heat_maps.append(g_z)
        return np.stack( heat_maps, axis=0 )



def plot_predictions(config, validation_image_batch, validation_label_batch, validation_output_batch, number_images_print):
    '''
    Plots a sample of images from the validation batch and the corresponding true key points (in cyan),
    labels for the true key points (in green), the predicted version of the key points (in red),
    and a yellow line between the predicted and true key point.
    '''

    # Config variables
    image_height = config.data_constants['IMAGE_HEIGHT']
    image_width = config.data_constants['IMAGE_WIDTH']
    num_points = config.data_constants['NUM_POINTS']
    kp_plot_rad = config.data_constants['KP_PLOT_RAD']



    # Check rows is less than validation set size
    if (number_images_print > len(validation_image_batch)):
        number_images_print = len(validation_image_batch)
        print('Warning: Attempted to print more images than the validation batch contains!')
        print('Only printing',number_images_print,'image(s).')
    sample_image_indices = random.sample(range(0, len(validation_image_batch)), number_images_print)
 
    # Plot Image With Key Points, Labels, Guesses for Key Points, Line between Each Actual KP and Guess KP
    for img_to_print_idx in sample_image_indices[:5]:
        plt.figure(figsize=(48,(image_height/image_width)*48))
        I = validation_image_batch[img_to_print_idx].to("cpu").type(torch.float32)
        L_heatmaps = (validation_label_batch[img_to_print_idx].to("cpu")).type(torch.float32)
        # Convert Heatmaps to Points By Taking The Maximum For Each Keypoint (Labels)
        L = []
        for kp_ind in range(0,L_heatmaps.shape[0]):
            Lm_ind = torch.argmax(L_heatmaps[kp_ind])
            Lx_ind = Lm_ind.item() % L_heatmaps[kp_ind].shape[1]
            Ly_ind = Lm_ind.item() // L_heatmaps[kp_ind].shape[1] # Indexed by top left corner (same as OpenCV)
            L.append([Lx_ind, Ly_ind])
        G_heatmaps = (validation_output_batch[img_to_print_idx].to("cpu")).type(torch.float32)
        # Convert Heatmaps to Points By Taking The Maximum For Each Keypoint (Guesses)
        G = []
        for kp_ind in range(0,G_heatmaps.shape[0]):
            Gm_ind = torch.argmax(G_heatmaps[kp_ind])
            Gx_ind = Gm_ind.item() % G_heatmaps[kp_ind].shape[1]
            Gy_ind = Gm_ind.item() // G_heatmaps[kp_ind].shape[1] # Indexed by top left corner (same as OpenCV)
            G.append([Gx_ind, Gy_ind])
        img_PIL = torchvision.transforms.ToPILImage()(I.type(torch.uint8))
        img_cv = cv2.cvtColor(np.array(img_PIL), cv2.COLOR_GRAY2BGR)
        for point_idx in range(0,num_points):
            cv2.circle(img_cv,((L[point_idx][0]),(L[point_idx][1])), kp_plot_rad, (0,255,255))
            cv2.circle(img_cv,((G[point_idx][0]),(G[point_idx][1])), kp_plot_rad, (255,0,0))
            #cv2.putText(img_cv,str(point_idx),(int(IMAGE_WIDTH*L[point_idx,0].item()),int(IMAGE_HEIGHT* (1 - L[point_idx,1].item()))),0,FONT_SCALE_PLOT,(0,255,0))
            cv2.line(img_cv, ((L[point_idx][0]),(L[point_idx][1])),\
                     ((G[point_idx][0]),(G[point_idx][1])), (255,255,0))
        plt.imshow(img_cv)
        plt.show()
        plt.imshow(L_heatmaps[0].numpy())
        plt.colorbar()
        plt.show()
        plt.imshow(G_heatmaps[0].detach().numpy())
        plt.colorbar()
        plt.show()
