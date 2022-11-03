import torch
import torch.nn as nn
import albumentations as A


class Configuration:
    def __init__(self):
        self.etl = \
                    {
                    'raw_data_file': '/blue/banks/nicholasverdugo/Shoulder-Keypoint/csv_data/sca.csv',
                    'processed_path': '/blue/banks/nicholasverdugo/Shoulder-Keypoint/processed',
                    'val_size':  0.2,
                    'test_size': 0.01,
                    'random_state': 42
                    }
        self.data_constants = \
                    {
                    'IMAGE_HEIGHT' : 1024,
                    'IMAGE_WIDTH' : 1024,
                    'DATA_HOME_DIR' : '/blue/banks/nicholasverdugo/Shoulder-Keypoint/',
                    'MODEL_TYPE' : 'points',

                    'IMAGES_GPU_DATA_TYPE' : torch.FloatTensor,
                    'LABELS_GPU_DATA_TYPE' : torch.FloatTensor,
                    'MAX_EPOCHS': 500,
                    'VALIDATION_BATCH_SIZE': 2,
                    
                    'IMAGE_THRESHOLD': 0,
                    'NUM_PRINT_IMAGE': 2,
                    'ALPHA_IMG' : 0.3,
                    'ALPHA_IMG' : 0.3,
                    'IMG_CHANNELS' : 1,
                    
                    'MODEL_NAME' : 'scapula',
                    'IMAGE_DIRECTORY' : '/blue/banks/JTML/SHOULDER_DATA/',
                    'LOAD_FROM_CHECKPOINT' : False,

                    'NUM_PRINT_IMG' : 1,
                    'KP_PLOT_RAD' : 3,

                    'NUM_POINTS' : 256,

                    'GAUSSIAN_STDEV_HEATMAP' : 5,
                    'GAUSSIAN_AMP' : 1e6,

                    'STORE_DATA_RAM' : False,

                    'CROP_IMAGES' : False,
                    'CROP_MIN_X' : 0.29,
                    'CROP_MAX_X' : 0.84,
                    'CROP_MIN_Y' : 0.45,
                    'CROP_MAX_Y' : 0.95,
                    
                    'images_per_grid' : 1,
                    'per_grid_image_count_height' : 1, 
                    'per_grid_image_count_width' : 1,

                    # I don't think we need these lines :/
                    # 'TEST_OUTPUT_DIR' : '',
                    # 'TEST_ORIGINAL_DIR' : ''

                    }
                    # NOTE: THESE NEED TO BE LOWERCASE FOR PyTorch DataLoader object
        self.data_loader_parameters = \
                    {
                    'batch_size' : 4,
                    'shuffle' : True,
                    'num_workers' : 0,
                    'pin_memory' : False
                    }
        self.train = \
                    {
                    'LOSS_FN' : nn.MSELoss()
                    }
        self.test = \
                    {
                    'CUSTOM_TEST_SET': False,
                    'TEST_SET_NAME' : 'NA'
                    }
        self.transform = A.Compose([
                                    A.RandomGamma(always_apply=False, p = 0.5,gamma_limit=(10,300)),
                                    A.ShiftScaleRotate(always_apply = False, p = 0.5,shift_limit=(-0.06, 0.06), scale_limit=(-0.1, 0.1), rotate_limit=(-180,180), interpolation=0, border_mode=0, value=(0, 0, 0)),
                                    A.Blur(always_apply=False, blur_limit=(3, 10), p=0.2),
                                    A.Flip(always_apply=False, p=0.5),
                                    A.InvertImg(always_apply=False, p=0.5),
                                    A.CoarseDropout(always_apply = False, p = 0.25, min_holes = 1, max_holes = 100, min_height = 25, max_height=25),
                                    A.MultiplicativeNoise(always_apply=False, p=0.25, multiplier=(0.1, 2), per_channel=True, elementwise=True)
                                    ], p=0.85)