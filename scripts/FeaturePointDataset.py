
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from skimage import io
import math
import numpy as np
from _____utility import create_gaussian_heatmap


class FeaturePointDataset(Dataset):
    """
    Feature Point Dataset.
    1. Grayscale images are stored in grids to reduce the number of file paths.
    2. Labels are read in as NUM_POINTS 2-tuples per image stored in a text file, then transformed to 2D Gaussian heat maps.
        a). Labels should be normalized in the text file already.
        b). The label text file is stored in the %data_home_dir%/labels/%model_type% folder and is always named
        '%model_type%_labels.txt'.
        c). Checks will be performed to ensure that each 1x1 grid in %data_home_dir%/%evaluation_type% has a corresponding label, 
        else the class will throw an exception.
    """

    
    def __init__(self, config, store_data_ram, evaluation_type, num_points, transform=None):
        """
        Args:
            data_constants (dictionary): Dictionary of vital constants about data.
            store_data_ram (boolean): Choose to store data in RAM for faster processing, or save RAM at a cost in performance.
            evaluation_type (string): Dataset evaluation type (must be 'training', 'validation', or 'test')
            num_points (int): Number of feature points per image.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        # Create a local copy of the configuration
        self.config = config
        
        # Local copy of num_points
        self.num_points = num_points
        
        # Check that evaluation_type is valid and then store
        if evaluation_type in ['training', 'validation', 'test']:
            self.evaluation_type = evaluation_type
        else:
            raise Exception('Incorrect evaluation type! Must be either \'training\', \'validation\', or \'test\'.')
        
        # Create full paths to all grids and their Labels
        '''
        Search %data_home_dir%/%evaluation_type%/ for all .tif files. 
        Check that properly named label text file in %data_home_dir%/labels/%model_type%/ exists for the given model type.
        Throw an exception if this is not the case.
        '''

        print('About to read in csv')        
        input_data_frame = pd.read_csv(self.config.etl['processed_path'] + '/' + config.data_constants['MODEL_NAME'] + '/train_' + config.data_constants['MODEL_NAME'] + '.csv', header=None, names=['grid', 'keypoints'])
        input_data_frame = input_data_frame.iloc[1:].reset_index(drop=True)  # MIGHT NEED TO REINDEX BECAUSE INDEX STARTS AT 1

        self.grids_fullpaths = input_data_frame['grid'].apply(lambda x: (self.config.data_constants['IMAGE_DIRECTORY'] + x[0:-1]))
        print('grids_fullpaths created')
        print(self.grids_fullpaths.iloc[0])

        # Calculate grid count
        self.grid_count = len(self.grids_fullpaths)

        self.label_point_data = np.vstack(input_data_frame['keypoints'].apply(lambda x: x.split(',')).apply(lambda x: np.array(x, dtype=float).reshape(-1,2)))  # USED TO WORK BUT DOES NOT NEED TO BE VSTACKED
        self.label_point_data = np.reshape(self.label_point_data, (input_data_frame.shape[0], num_points, 2))
        
        if self.label_point_data.shape != (len(self.grids_fullpaths),num_points,2):
                     raise Exception('Error, label data array has shape ' + str(self.label_point_data.shape) + '!')
        # Optional transform
        self.transform = transform
        print('keypoints loaded in')
        # Store image tensors and label tensors to CPU RAM option (should be faster as long as there is room in the RAM)
        # We probably don't have enough RAM on hipergator to use this option. set to false in config to avoid out of memory error.
        self.store_data_ram = store_data_ram
        print('self.store_data_ram')
        self.data_storage = []
        if self.store_data_ram:
            for idx in range(self.grid_count*self.config.data_constants['images_per_grid']): # Total number of images # THIS IS THE REAL FOR LOOP THAT GETS ALL IMAGES
                self.data_storage.append(self.read_in_data(idx))
                print("image ", idx, " processed")
                
        # Print successful initialization message
        print ("Successfully initialized " + self.evaluation_type + ' dataset!')

    def __len__(self):
        return self.grid_count*self.config.data_constants['images_per_grid'] # Total number of images in data type (n) # FOR REAL USE
    
    def read_in_data(self, idx):
        # Read in image grid
        grid_idx = idx//self.config.data_constants['images_per_grid']
        grid_image = io.imread(self.grids_fullpaths[grid_idx], as_gray=True)
        
        # Extract image from grid using top-left to bottom-right ordering
        idx_in_grid = idx%self.config.data_constants['images_per_grid']
        img_top_row_idx = (idx_in_grid//self.config.data_constants['per_grid_image_count_width'])*self.config.data_constants['IMAGE_HEIGHT']
        img_left_col_idx = (idx_in_grid%self.config.data_constants['per_grid_image_count_width'])*self.config.data_constants['IMAGE_WIDTH']
        image = grid_image[img_top_row_idx:img_top_row_idx + self.config.data_constants['IMAGE_HEIGHT'],\
                          img_left_col_idx:img_left_col_idx + self.config.data_constants['IMAGE_WIDTH']]
        
        # Label should always be in [0,1] format when read in and then transformed into Gaussian heatmap       
        # In PyTorch, images are represented as [channels, height, width] so must add 1 channel dimension
        if self.config.data_constants['CROP_IMAGES']:
            LEFT_X_PIX = math.floor(self.config.data_constants['CROP_MIN_X']*self.config.data_constants['IMAGE_WIDTH'])
            RIGHT_X_PIX = math.ceil(self.config.data_constants['CROP_MAX_X']*self.config.data_constants['IMAGE_WIDTH'])
            LEFT_X_PIX -= (32 - ((RIGHT_X_PIX - LEFT_X_PIX) % 32)) # So Width is Divisible by 32
            BOT_Y_PIX = math.floor(self.config.data_constants['CROP_MIN_Y']*self.config.data_constants['IMAGE_HEIGHT'])
            TOP_Y_PIX = math.ceil(self.config.data_constants['CROP_MAX_Y']*self.config.data_constants['IMAGE_HEIGHT'])
            BOT_Y_PIX -= (32 - ((TOP_Y_PIX - BOT_Y_PIX) % 32)) # So Height is Divisible by 32
            image = torch.ByteTensor(image[None, (self.config.data_constants['IMAGE_HEIGHT'] - TOP_Y_PIX):(self.config.data_constants['IMAGE_HEIGHT']-BOT_Y_PIX), LEFT_X_PIX:RIGHT_X_PIX]) # Store as byte (to save space) then convert
            cropped_point_list = [] # Need to Rescale Points
            for orig_point in self.label_point_data[idx]:
                cropped_point_list.append([(orig_point[0]*self.config.data_constants['IMAGE_WIDTH'] - LEFT_X_PIX)/(RIGHT_X_PIX-LEFT_X_PIX),\
                                           (orig_point[1]*self.config.data_constants['IMAGE_WIDTH'] - BOT_Y_PIX)/(TOP_Y_PIX-BOT_Y_PIX)])
            label = torch.FloatTensor(create_gaussian_heatmap(self.config, cropped_point_list, TOP_Y_PIX-BOT_Y_PIX, RIGHT_X_PIX-LEFT_X_PIX))
            #image = torch.zeros([1,TOP_Y_PIX-BOT_Y_PIX, RIGHT_X_PIX-LEFT_X_PIX],dtype=torch.uint8)
            #label = torch.zeros([NUM_POINTS,TOP_Y_PIX-BOT_Y_PIX, RIGHT_X_PIX-LEFT_X_PIX], dtype=torch.float)
        else:
            image = torch.ByteTensor(image[None, :, :]) # Store as byte (to save space) then convert when called in __getitem__
            label = torch.FloatTensor(create_gaussian_heatmap(self.config, self.label_point_data[idx], self.config.data_constants['IMAGE_HEIGHT'], self.config.data_constants['IMAGE_WIDTH']))
        
        # Form sample and transform if necessary
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __getitem__(self, idx):
        if self.store_data_ram:
            return {'image': self.data_storage[idx]['image'].type(self.config.data_constants['IMAGES_GPU_DATA_TYPE']), 'label':\
                   self.data_storage[idx]['label']}
        else:
            # Read in image grid
            grid_idx = idx//self.config.data_constants['images_per_grid']
            grid_image = io.imread(self.grids_fullpaths[grid_idx], as_gray=True)

            # Extract image from grid using top-left to bottom-right ordering
            idx_in_grid = idx%self.config.data_constants['images_per_grid']
            img_top_row_idx = (idx_in_grid//self.config.data_constants['per_grid_image_count_width'])*self.config.data_constants['IMAGE_HEIGHT']
            img_left_col_idx = (idx_in_grid%self.config.data_constants['per_grid_image_count_width'])*self.config.data_constants['IMAGE_WIDTH']
            image = grid_image[img_top_row_idx:img_top_row_idx + self.config.data_constants['IMAGE_HEIGHT'],\
                              img_left_col_idx:img_left_col_idx + self.config.data_constants['IMAGE_WIDTH']]
            
            # Label should always be in [0,1] format when read in and then transformed into Gaussian heatmap       
            # In PyTorch, images are represented as [channels, height, width] so must add 1 channel dimension
            if self.config.data_constants['CROP_IMAGES']:
                LEFT_X_PIX = math.floor(self.config.data_constants['CROP_MIN_X']*self.config.data_constants['IMAGE_WIDTH'])
                RIGHT_X_PIX = math.ceil(self.config.data_constants['CROP_MAX_X']*self.config.data_constants['IMAGE_WIDTH'])
                LEFT_X_PIX -= (32 - ((RIGHT_X_PIX - LEFT_X_PIX) % 32)) # So Width is Divisible by 32
                BOT_Y_PIX = math.floor(self.config.data_constants['CROP_MIN_Y']*self.config.data_constants['IMAGE_HEIGHT'])
                TOP_Y_PIX = math.ceil(self.config.data_constants['CROP_MAX_Y']*self.config.data_constants['IMAGE_HEIGHT'])
                BOT_Y_PIX -= (32 - ((TOP_Y_PIX - BOT_Y_PIX) % 32)) # So Height is Divisible by 32
                image = torch.ByteTensor(image[None, (self.config.data_constants['IMAGE_HEIGHT'] - TOP_Y_PIX):(self.config.data_constants['IMAGE_HEIGHT']-BOT_Y_PIX), LEFT_X_PIX:RIGHT_X_PIX]) # Store as byte (to save space) then convert
                cropped_point_list = [] # Need to Rescale Points
                for orig_point in self.label_point_data[idx]:
                    cropped_point_list.append([(orig_point[0]*self.config.data_constants['IMAGE_WIDTH'] - LEFT_X_PIX)/(RIGHT_X_PIX-LEFT_X_PIX),\
                                               (orig_point[1]*self.config.data_constants['IMAGE_HEIGHT'] - BOT_Y_PIX)/(TOP_Y_PIX-BOT_Y_PIX)])
                label = torch.FloatTensor(create_gaussian_heatmap(self.config, self.cocropped_point_list, TOP_Y_PIX-BOT_Y_PIX, RIGHT_X_PIX-LEFT_X_PIX))
            else:
                image = torch.FloatTensor(image[None, :, :]) # Store as byte (to save space) then convert when called in __getitem__
                label = torch.FloatTensor(create_gaussian_heatmap(self.config, self.label_point_data[idx], self.config.data_constants['IMAGE_HEIGHT'], self.config.data_constants['IMAGE_WIDTH']))
        
            # Form sample and transform if necessary
            sample = {'image': image, 'label': label}
            if self.transform:
                sample = self.transform(sample)
            return sample