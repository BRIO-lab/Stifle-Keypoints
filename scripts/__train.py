import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
import sys
import os
import math
from pose_hrnet import PoseHighResolutionNet
import pandas as pd
import click
from _____utility import plot_predictions, set_logger
from FeaturePointDataset import FeaturePointDataset
from importlib import import_module


@click.command()
@click.argument("config_file", type=str, default="config")
def train(config_file):

    print("training begun!")
    
    ########################
    # Load in configuration file
    ########################
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    
    config_dir = os.getcwd() + "/config/"
    sys.path.append(config_dir)
    config_module = import_module(config_file)
    config = config_module.Configuration()

    """
    Set up logger and define training and validation sets
    """

    dir_name = config.data_constants["MODEL_NAME"]
    if os.path.isdir("log/" + dir_name) == False:
        os.os.mkdir("log/" + dir_name)
    
    logger = set_logger(os.path.join("log",dir_name,"train_"+config.data_constants["MODEL_NAME"]+".log"))
    logger.info(f"Load config from {config_file}")
    logger.info(f"GPU Devices available: {torch.cuda.device_count()}")
    train_name = "train_" + config.data_constants["MODEL_NAME"] + ".csv"
    val_name   = "val_" + config.data_constants["MODEL_NAME"] + ".csv"
   

    train_path = os.path.join(config.etl["processed_path"],config.data_constants["MODEL_NAME"],train_name)
    val_path  = os.path.join(config.etl["processed_path"],config.data_constants["MODEL_NAME"],val_name)

    train = pd.read_csv(train_path)
    train = train.to_numpy
    
    logger.info(f"Training set loaded from {train_path}")
    
    validation = pd.read_csv(val_path)
    validation = validation.to_numpy
    logger.info(f"Validation set loaded from {val_path}")


    num_points = config.data_constants['NUM_POINTS']
    img_channels = config.data_constants['IMG_CHANNELS']
    load_from_checkpoint = config.data_constants['LOAD_FROM_CHECKPOINT']
    model_name = config.data_constants['MODEL_NAME']


    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    logger.info('Is cuda available:  ' +  str(use_cuda))
    main_device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True # Eventually makes faster after initial computational cost
    logger.info('Main device:   ' + str(main_device))

    print("creating model...")
    
    # Make model and move to main_device
    model = PoseHighResolutionNet(num_key_points = num_points, num_image_channels = img_channels)
    model = model.to(main_device)
 
    print("model created!")
 
    additional_devices = []

    # Split Across Multiple Devices
    if torch.cuda.device_count() > 1 and main_device.type == 'cuda':
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model) # Splits batches across GPUs
        # Name additional devices
        for device_id in range(1,torch.cuda.device_count()):
            additional_devices.append(torch.device("cuda:" + str(device_id)))
    """ 
    # Print info when using cuda
    if main_device.type == 'cuda':
        print(torch.cuda.get_device_name(main_device),main_device,'- Main Device')
        print('Memory Usage:')
        print('Allocated:', str(round(torch.cuda.memory_allocated(main_device)/1024**3,1)), 'GB')
        print('Cached:   ', str(round(torch.cuda.memory_reserved(main_device)/1024**3,1)), 'GB')                              
        for extra_dev in additional_devices:
            print(torch.cuda.get_device_name(extra_dev),extra_dev)
            print('Memory Usage:')
            print('Allocated:', str(round(torch.cuda.memory_allocated(extra_dev)/1024**3,1)), 'GB')
            print('Cached:   ', str(round(torch.cuda.memory_reserved(extra_dev)/1024**3,1)), 'GB')                              
    """
    # Load model weights if loading from previous checkpoint
    if load_from_checkpoint:
        checkpoint = torch.load('./'+ model_name + '.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info('Checkpoint average validation loss:',checkpoint['validation_loss'])
    else:
        start_epoch = -1
        
    # Make loss function and move to device (will run the cuda loss function if input tensor is a cuda tensor, but just in case)
    loss_fn = torch.nn.MSELoss().to(main_device)
    
    # Make Adam optimizer
    # It is recommended to move a model to GPU before constructing an optimizer:
    # (https://pytorch.org/docs/master/optim.html)
    optimizer = torch.optim.Adam(model.parameters())
    if load_from_checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Training Generators
    training_set = FeaturePointDataset(config, config.data_constants['STORE_DATA_RAM'], 'training', config.data_constants['NUM_POINTS'])

    training_generator = torch.utils.data.DataLoader(training_set, **config.data_loader_parameters)  # NEW - Arman
    
     # Validation Generators
    validation_set = FeaturePointDataset(config, config.data_constants['STORE_DATA_RAM'], 'validation', config.data_constants['NUM_POINTS'])
    validation_generator = torch.utils.data.DataLoader(validation_set, **config.data_loader_parameters)


    # Initialize current minimums
    if config.data_constants['LOAD_FROM_CHECKPOINT']:
        current_minimum_avg_validation_loss = checkpoint['validation_loss']
    else:
        current_minimum_avg_validation_loss = math.inf
    current_minimum_epoch_idx = 0
    
    # Loop over epochs
    for epoch_idx in range(start_epoch + 1, config.data_constants['MAX_EPOCHS']):
        # Epoch timer
        epoch_timer_start = time.time()
        
        # Training loop for an epoch
        logger.info("-"*65)
        logger.info('BEGIN EPOCH', epoch_idx)
        model.train()
        for training_batch_idx, training_batch in enumerate(training_generator):
            # Transfer to GPU
            training_batch, training_batch_labels = training_batch['image'].to(main_device, dtype=torch.float, non_blocking=True)\
            ,training_batch['label'].to(main_device, dtype=torch.float, non_blocking=True)
            
            # Zero the gradient buffers
            optimizer.zero_grad()
            
            # Forward through the model
            training_output = model(training_batch) # NOT NORMALIZING GRAYSCALE
            
            # Calculate loss
            training_loss = loss_fn(training_output, training_batch_labels)
            logger.info(f'Epoch {epoch_idx} --- Batch  {training_batch_idx} training loss before optimization step: {training_loss.item()}')
                
            # Backprop then update using optimizer
            training_loss.backward()
            optimizer.step()
        
        # Calculate average loss over validation set
        total_validation_loss = 0
        model.eval()
        for validation_batch in validation_generator:
            # Transfer to GPU
            validation_batch, validation_batch_labels = validation_batch['image'].\
            to(main_device, dtype=torch.float, non_blocking=True),\
            validation_batch['label'].to(main_device, dtype=torch.float, non_blocking=True)
            
            # Forward through the model
            validation_output = model(validation_batch) # NOT NORMALIZING GRAYSCALE
            
            # Calculate loss and keep a running sum total of the loss over all the validation data
            validation_loss = loss_fn(validation_output, validation_batch_labels)
            total_validation_loss += validation_loss.item()*len(validation_batch)
        average_validation_loss = total_validation_loss/len(validation_set)
        '''
        If the average validation loss is less than the current minimum, update the current minimum.
        Also, print out original image/predicted segmentation image/overlay of pairs for a few samples from the validation set.
        Finally, save the model's learned parameters to the SSD.
        '''
        if average_validation_loss < current_minimum_avg_validation_loss:
            current_minimum_avg_validation_loss = average_validation_loss
            current_minimum_epoch_idx = epoch_idx
            logger.info(f'New minimum average validation loss found:  {current_minimum_avg_validation_loss}')
            
            # Print out image, predicted segmentation mask, and an overlay
            if epoch_idx == 0 or epoch_idx == config.data_constants['MAX_EPOCHS']:  # ADDED BY ARMAN ONLY PRINT FIRST AND LAST EPOCH
                plot_predictions(config, validation_batch,validation_batch_labels, validation_output,config.data_constants['NUM_PRINT_IMG'])
            
            # Save the model
            torch.save({'epoch': epoch_idx,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()\
                        ,'validation_loss': current_minimum_avg_validation_loss}, './'+ config.data_constants['MODEL_NAME'] + '.pth')
        elif config.data_constants['LOAD_FROM_CHECKPOINT'] and epoch_idx == (start_epoch + 1): # Print out on first epoch if loading from a checkpoint
            # Print out image, predicted segmentation mask, and an overlay
            plot_predictions(config, validation_batch, validation_batch_labels, validation_output,config.data_constants['NUM_PRINT_IMG'])
            current_minimum_epoch_idx = start_epoch

        logger.info(f'Epoch {epoch_idx} has average validation loss: {average_validation_loss}')
        logger.info(f'Current minimum average validation loss:  {current_minimum_avg_validation_loss} (from epoch {str(current_minimum_epoch_idx)} )')
        logger.info(f'Epoch {epoch_idx} runtime: {time.time() - epoch_timer_start} seconds')


if __name__ == "__main__":
    train()