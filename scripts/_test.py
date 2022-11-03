import torch
import torch.nn as nn
import sys
import os
from pose_hrnet import PoseHighResolutionNet
import numpy as np
import click
from FeaturePointDataset import FeaturePointDataset
from _____utility import plot_predictions
from importlib import import_module

@click.command()
@click.argument("config_file", type=str, default="config")
def test(config_file):
    config_dir = os.getcwd() + "/config/"
    sys.path.append(config_dir)
    config_module = import_module(config_file)
    config = config_module.Configuration()

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    main_device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True # Eventually makes faster after initial computational cost
    
    # Make model and move to main_device
    model = PoseHighResolutionNet(num_key_points = config.data_constants['NUM_POINTS'], num_image_channels = config.data_constants['IMG_CHANNELS'])
    model = model.to(main_device)

    # Name additional devices
    additional_devices = []
    # Split Across Multiple Devices
    if torch.cuda.device_count() > 1 and main_device.type == 'cuda':
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model) # Splits batches across GPUs
        for device_id in range(1,torch.cuda.device_count()):
            additional_devices.append(torch.device("cuda:" + str(device_id)))
    """
    # Print info when using cuda
    if main_device.type == 'cuda':
        print(torch.cuda.get_device_name(main_device),main_device,'- Main Device')
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(main_device)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(main_device)/1024**3,1), 'GB')                              
        for extra_dev in additional_devices:
            print(torch.cuda.get_device_name(extra_dev),extra_dev)
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(extra_dev)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(extra_dev)/1024**3,1), 'GB')                              
    """
    # Load model weights if loading from previous checkpoint
    if config.data_constants['LOAD_FROM_CHECKPOINT']:
        checkpoint = torch.load('./'+ config.data_constants['MODEL_NAME'] + '.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print('Checkpoint average validation loss:',checkpoint['validation_loss'])
    else:
         raise Exception('Must be loading trained model if running in test mode!')
            
     # Make loss function and move to device (will run the cuda loss function if input tensor is a cuda tensor, but just in case)
    loss_fn = torch.nn.MSELoss().to(main_device)
        
     # Test Generators
    test_set = FeaturePointDataset(config.data_constants['STORE_DATA_RAM'], 'test', config.data_constants['NUM_POINTS'])
    test_generator = torch.utils.data.DataLoader(test_set, batch_size=config.data_constants['VALIDATION_BATCH_SIZE'], shuffle = False)
    
    '''
    BEGIN READ IN IMAGE INFORMATION:
    Width, Height, True Pose [X,Y,Z,ZA,XA,YA], Calibration File Info [PD,XO,YO,PP], 
    Model Key Points [NUM_POINTS by [X,Y,Z]], Projected Key Points [NUM_POINTS by [X,Y]]
    
    # Read In Each Line of Info File
    info_file = TEST_OUTPUT_DIR + '/info_' + MODEL_TYPE +'.txt'
    info_lines = [line.rstrip('\n') for line in open(info_file)]
    # Double Check That There Are 13*# of Test Files)
    if len(info_lines) != 13*len(test_set):
        raise Exception('Info file length is not 13 x the number of loaded test files!')

    # Read in Projected Key Points File
    # Projected Key Points
    proj_kp_file = TEST_OUTPUT_DIR + '/' + MODEL_TYPE + '_KPlabels.txt'
    proj_kp_lines = [line.rstrip('\n') for line in open(proj_kp_file)]
    # Double Check That There Are NUM_POINTS + 1*# of Test Files)
    if len(proj_kp_lines) != (NUM_POINTS + 1)*len(test_set):
        raise Exception('KP file length is not (NUM_POINTS + 1) x the number of loaded test files!')

    # For Each Image, Create Dictionary with The Image's:
    # Width, Height, True Pose [X,Y,Z,ZA,XA,YA], Calibration File Info [PD,XO,YO,PP], 
    # Model Key Points [NUM_POINTS by [X,Y,Z]], Projected Key Points [NUM_POINTS by [X,Y]]
    filtered_image_info = []
    print('Test set has', len(test_set),'images!')
    for image_indx in range(len(test_set)):
        # Image Width and Height
        img_w = int(info_lines[image_indx*13 + 9].split(':')[1].strip())
        img_h = int(info_lines[image_indx*13 + 10].split(':')[1].strip())
        # True Pose
        true_pose_str = info_lines[image_indx*13 + 11].split(':')[1].strip().split(',')
        true_pose = [float(i) for i in true_pose_str]
        # Get Image Home Directory
        img_home_dir = os.path.dirname(info_lines[image_indx*13].split('Path:')[1].strip())
        # Calibration Info
        calib_info_str = [line.strip() for line in open(img_home_dir + '/calibration.txt')][1:5]
        calib_info = [float(i) for i in calib_info_str]
        # Model Key Points (NUM_POINTS by (X,Y,Z))
        model_kp_whole_file = [line.rstrip('\n') for line in open(img_home_dir + '/'+ MODEL_TYPE +'.kp')]
        # Check there are 2 + NUM_POINTS LINES
        if len(model_kp_whole_file) != 2 + NUM_POINTS:
            raise Exception('Model key points file is not the right length!')
        model_kp_str = [line.split(':')[1].strip().split(',') for line in model_kp_whole_file[1:NUM_POINTS+1]]
        model_kp = [[float(j) for j in i] for i in model_kp_str]
        # Projected Key Points
        proj_kp_str = [line.strip().split(',') for line in proj_kp_lines[image_indx*(NUM_POINTS+1) + 1:\
                                                                          (image_indx*(NUM_POINTS+1) + NUM_POINTS + 1)]]
        proj_kp = [[float(j) for j in i] for i in proj_kp_str]
        # Store in Dictionary
        filtered_image_info.append({'image_width': img_w, 'image_height': img_h,'true_pose': true_pose,'calibration': calib_info,\
                                 'model_kp': model_kp, 'projected_kp': proj_kp})
    
    
    END READ IN IMAGE INFORMATION
    '''
    # Loop: predict, print and save
    model.eval()
    images_saved = 0
    total_test_loss = 0
    output_storage = np.empty([len(test_set), config.data_constants['NUM_POINTS'], 2])
    test_index_counter = 0
    for test_batch in test_generator:
        # Transfer to GPU
        test_batch, test_batch_labels = test_batch['image'].\
        to(main_device, dtype=torch.float, non_blocking=True),\
        test_batch['label'].to(main_device, dtype=torch.float, non_blocking=True)

        # Forward through the model
        test_output = model(test_batch)
        
        # Calculate loss and keep a running sum total of the loss over all the test data
        test_loss = loss_fn(test_output, test_batch_labels)
        total_test_loss += test_loss.item()*len(test_batch)
        
        # Print out image, predicted segmentation mask, and an overlay
        plot_predictions(test_batch,test_batch_labels, test_output, config.data_constants['VALIDATION_BATCH_SIZE'])

        # Store Outputs
        # Convert Heatmaps to Points By Taking The Maximum For Each Keypoint (Labels)
        heatmaps_output = test_output.cpu().detach()
        for tb_ind in range(len(test_batch)):
            KP_loc_guess = []
            for kp_ind in range(config.data_constants['NUM_POINTS']):
                m_ind = torch.argmax(heatmaps_output[tb_ind][kp_ind])
                x_ind = m_ind.item() % heatmaps_output[tb_ind][kp_ind].shape[1]
                y_ind = m_ind.item() // heatmaps_output[tb_ind][kp_ind].shape[1] # Indexed by top left corner (same as OpenCV)
                KP_loc_guess.append([x_ind/config.data_constants['IMAGE_WIDTH'], (config.data_constants['IMAGE_HEIGHT'] - y_ind)/config.data_constants['IMAGE_HEIGHT']])
            output_storage[test_index_counter+tb_ind,:,:] = KP_loc_guess
         
        test_index_counter += len(test_batch)
    # Print out average test batch loss
    average_test_loss = total_test_loss/len(test_set)
    print('Average loss over test set:',average_test_loss)
    return 0


if __name__ == "__main__":
    test()
