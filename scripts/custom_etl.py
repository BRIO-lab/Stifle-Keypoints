"""
Sasank Desaraju
3/8/2023

This is a custom etl file to create the TTV CSV files with a naive set.
This also uses the big_data CSV files which makes it easier to isolate specific patients etc.
This is assuming there are two models for fem and tib, respectively, and that they are using the same dataset. Wait, just literally use the same dataset for both models. That's fine.
"""

import sys
import os
import time
from sklearn.model_selection import train_test_split as tts
import pandas as pd
import numpy as np
import glob
import csv

# * Set the parameters
#ROOT_DIR = os.getcwd()
ROOT_DIR = '/home/sasank/Documents/GitRepos/Stifle-Keypoints/'
RAW_DATA_FILE = os.path.join(ROOT_DIR, 'data', 'pruned_ISTA.csv')
ETL_NAME = 'ISTA_Split'
NAIVE_PATIENT_NUMBER = 9
TEST_SIZE = 0.2
VAL_SIZE = 0.2

"""
Plan
Load the raw data file
Remove the naive dataset
Split the remaining data into train, val, and test
Write the 4 CSVs to the data/ETL_NAME directory
"""

# * Load the raw data file
raw_data = pd.read_csv(RAW_DATA_FILE)
print('Raw data shape: ', raw_data.shape)

# ! Remove Patient 3 Session 1 because the ground truth is wrong - we handled this by removing the data from the raw data file using PnP-Solver/big_data/prune_data.py
#raw_data.drop(raw_data[(raw_data['Patient number'] == 3) & (raw_data['Session number'] == 1)].index, inplace=True)

#raw_data.to_csv(os.path.join(ROOT_DIR, 'data', 'asdf_proc_64KP_data.csv'), index=False)

print(raw_data.shape)

#sys.exit()

# * Remove the naive dataset
naive_data = raw_data[raw_data['Patient number'] == NAIVE_PATIENT_NUMBER]
raw_data = raw_data[raw_data['Patient number'] != NAIVE_PATIENT_NUMBER]

# * Split the remaining data into train, val, and test
train_val_data, test_data = tts(raw_data, test_size=TEST_SIZE, random_state=42)
train_data, val_data = tts(train_val_data, test_size=VAL_SIZE, random_state=42)

# * Make sure the folder exists
if not os.path.exists(os.path.join(ROOT_DIR, 'data', ETL_NAME)):
    os.makedirs(os.path.join(ROOT_DIR, 'data', ETL_NAME))

# * Write the 4 CSVs to the data/ETL_NAME directory
train_data.to_csv(os.path.join(ROOT_DIR, 'data', ETL_NAME, 'train_' + ETL_NAME + '.csv'), index=False)
val_data.to_csv(os.path.join(ROOT_DIR, 'data', ETL_NAME, 'val_' + ETL_NAME + '.csv'), index=False)
test_data.to_csv(os.path.join(ROOT_DIR, 'data', ETL_NAME, 'test_' + ETL_NAME + '.csv'), index=False)
naive_data.to_csv(os.path.join(ROOT_DIR, 'data', ETL_NAME, 'naive_' + ETL_NAME + '.csv'), index=False)
