"""
Sasank Desaraju
9/22/22
"""

"""
I want to make it so that it will auto-create TVT .csv files if the raw data file is given as -1
"""

from datetime import datetime
from genericpath import isfile
from importlib import import_module
from unicodedata import name
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
#import click
import sys
import os
import time
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split as tts
import numpy as np
import glob
import csv

# parsing the config
CONFIG_DIR = os.getcwd() + '/config/'
sys.path.append(CONFIG_DIR)
config_module = import_module(sys.argv[1])
#config_module = import_module('config/config')
config = config_module.Configuration()


# setting up the logger
os.environ["WANDB_RUN_GROUP"] = config.init['WANDB_RUN_GROUP']
wandb_run = wandb.init(
    project=config.init['PROJECT_NAME'],
    name=config.init['RUN_NAME'],
    job_type='etl'
)


# Setting some paths
DATA_DIR = config.etl['DATA_DIR']
KP_DIR = config.etl['KEYPOINT_DIRECTORY'] + '/'
KP_PATH = os.getcwd() + '/' + KP_DIR
MODEL_DIR_NAME = config.init['MODEL_NAME']
PROCESSED_PATH = os.getcwd() + '/' + DATA_DIR + '/' + MODEL_DIR_NAME + '/'        # This is the path the .csv files will be saved to
if os.path.isdir(PROCESSED_PATH) == False:
    os.mkdir(PROCESSED_PATH)

# Create raw data file from image directory if no .csv is provided
RAW_DATA_FILE = config.etl['RAW_DATA_FILE']
if RAW_DATA_FILE == -1:
    #IMAGE_DIR = config.datamodule['IMAGE_DIRECTORY']
    CSV_NAME = config.init['MODEL_NAME']
    if os.path.isfile(PROCESSED_PATH + CSV_NAME + '.csv') == True:
        raise Exception('Error, the full data CSV is already present. Please set config.etl[\'RAW_DATA_FILE\'] to this file.')


    with open(PROCESSED_PATH + '/' + CSV_NAME + '.csv', 'w') as c:
        csvWriter = csv.writer(c)

        header = ['grid', config.dataset['MODEL_TYPE'] + '_kps']

        csvWriter.writerow(header)

        for file in glob.glob(KP_PATH + '*' + config.dataset['MODEL_TYPE'] + '*.txt'):
            # File reader
            f = open(file, 'r')

            # Save the first grid name to the first column
            # d for dictionary
            d = [f.readline()]

            # String that will be populated with the keypoints
            # k for keypoints
            k = ''

            # Write the keypoints to the string
            # When a new grid image is reached, write the keypoints to the csv and start again
            for line in f:
                if 'tif' in line:
                    # Strip the comma from the last keypoint line
                    k = k[:-1]
                    # Append the keypoints to the dictionary
                    d.append(k)
                    # Write the dictionary to the csv
                    csvWriter.writerow(d)
                    
                    # Reset the dictionary and keypoints
                    d = [line]
                    k = ''
                else:
                    k += line + ','

    RAW_DATA_FILE = PROCESSED_PATH + '/' + CSV_NAME + '.csv'






    """
    grid = np.array(glob.glob1(IMAGE_DIR, "grid*.tif"))
    grid = np.insert(grid, [0], ["img"])
    fem = np.array(glob.glob1(IMAGE_DIR, "fem*.tif"))
    fem = np.insert(fem, [0], ["fem"])
    tib = np.array(glob.glob1(IMAGE_DIR, "tib*.tif"))
    tib = np.insert(tib, [0], ["tib"])

    fem_and_tib = np.concatenate((fem[:,None],tib[:,None]), axis = 1)
    all_images_and_masks = np.concatenate((grid[:,None],fem_and_tib), axis = 1)
    headers = np.array([["img","fem","tib"]])
    np.savetxt(OUTPUT_DIR + '/' + CSV_NAME + '.csv', all_images_and_masks, fmt = '%s', delimiter = ',')
    """


# now assume the RAW_DATA_FILE lives at RAW_DATA_FILE
# read in data file
full_data = pd.read_csv(RAW_DATA_FILE)

# TTV split of data
train_and_val,test = tts(full_data,
                        test_size=config.etl['TEST_SIZE'],
                        random_state = config.etl['RANDOM_STATE'])
train,val  = tts(train_and_val,
                test_size=config.etl['VAL_SIZE'],
                random_state = config.etl['RANDOM_STATE'])

# name the .csv files
train_name = "train_" + config.init["MODEL_NAME"] + ".csv"
val_name =   "val_" + config.init["MODEL_NAME"] + ".csv"
test_name =  "test_" + config.init["MODEL_NAME"] + ".csv"


# export data
train.to_csv(PROCESSED_PATH + '/' + train_name, index=False)
test.to_csv(PROCESSED_PATH + '/' + test_name, index=False)
val.to_csv(PROCESSED_PATH + '/' + val_name, index = False)


