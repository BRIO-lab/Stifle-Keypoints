# let's try summ out


import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from monai.networks.nets import SwinUNETR
from kornia.geometry.subpix import SpatialSoftArgmax2d
import pytorch_lightning as pl
import wandb

import time
import nvtx

from loss import res_kp_loss
from utility import *

class KeypointNetModule(pl.LightningModule):
    def __init__(self, config, wandb_run, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters("learning_rate")
        self.config = config    
        #print("Pose ResNet is on device " + str(next(self.pose_hrnet.parameters()).get_device()))     # testing line
        #print("Is Pose ResNet on GPU? " + str(next(self.pose_hrnet.parameters()).is_cuda))            # testing line
        self.wandb_run = wandb_run
        self.loss_fn = res_kp_loss(
            gaussian_amp=self.config.dataset['GAUSSIAN_AMP'],
            gaussian_sigma=self.config.dataset['GAUSSIAN_STDDEV'],
            weighted_loss=self.config.loss['WEIGHTED_LOSS']
            )

        # ! Good grief what a hack. This is only for local testing.
        #WRITE_CSV = '/home/sasank/Documents/GitRepos/PnP-Solver/kp_estimates/naive_Ten_Dogs_64KP_estimates_pr.csv'
        #self.csv_file = pd.read_csv(WRITE_CSV)
        
        self.in_channels = self.config.dataset['IMG_CHANNELS']
        self.num_keypoints = self.config.dataset['NUM_KEY_POINTS']
        self.batch_size = self.config.datamodule['BATCH_SIZE']      # This is used in the logging steps in validation_step and test_step.
        self.image_height = self.config.dataset['IMAGE_HEIGHT']     # These two, too.
        self.image_width = self.config.dataset['IMAGE_WIDTH']

        self.my_dict = nn.ModuleDict({})

        # ******** Resnet 152 Architecture ********

        #"""
        self.my_dict["pre_block"] = nn.Sequential(
            nn.Conv2d(self.in_channels, 3, kernel_size=1, stride=1, padding=0, bias=False),     # This is just to convert the input channels to 3, which is what the resnet expects.
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        
        #self.my_dict["resnet"] = make_resnet(3, 64, 3, 1)
        # this 
        #self.my_dict["resnet"] = torchvision.models.resnet34(weights='IMAGENET1K_V1')
        # ! Disabling this for now to try SwinUNETR
        self.my_dict["resnet"] = torchvision.models.resnet152(weights='IMAGENET1K_V2')

        #assert self.num_keypoints <= 50, "If num_keypoints > 50, the last linear layer is growing bigger, which seems unreasonable."
        # The above assertion does not need to be made because it could be that the model is just finding a lower-dimensional representation of the keypoints, which, if accurate, would be a good thing.
        # ! If model performance is bad is bad when using 64 keypoints, then it may be a good idea to reexamine this.

        self.my_dict["keypoints"] = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 2 * self.num_keypoints)
        )
        #"""
    
        # ******** End Resnet 152 Architecture ********

        # ******** SwinUNETR Architecture ********

        """
        self.my_dict["swinunetr"] = SwinUNETR(
            img_size=(self.config.dataset['IMAGE_HEIGHT'], self.config.dataset['IMAGE_WIDTH']),
            in_channels=self.config.dataset['IMG_CHANNELS'],
            out_channels=self.config.dataset['NUM_KEY_POINTS'],
            spatial_dims=2
        )

        self.my_dict["kornia_keypoints"] = SpatialSoftArgmax2d(temperature=torch.tensor(1.0), normalized_coordinates=False)

        #self.my_dict["swinunetr_pipeline"] = nn.Sequential(
        #    swinunetr,
        #    kornia_spatial_soft_argmax
        #)
        """
        
        # ******** End SwinUNETR Architecture ********

        
        

    def forward(self, x):
        """This performs a forward pass on the dataset.

        Args:
            x (torch.Tensor): This is a tensor containing the input data.

        Returns:
            the forward pass of the dataset.
        """

        # * Resnet 152
        #"""
        x = self.my_dict["pre_block"](x)
        x = self.my_dict["resnet"](x)
        x = self.my_dict["keypoints"](x)
        #"""

        # * SwinUNETR
        """
        x = self.my_dict["swinunetr"](x)
        x = self.my_dict["kornia_keypoints"](x)
        """

        #print("x shape: " + str(x.shape))     # testing line

        # Reshape keypoints to be (batch_size, num_keypoints, 2)
        keypoints = x.view(-1, self.num_keypoints, 2)
        #print("keypoints shape: " + str(keypoints.shape))
        return keypoints

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters, lr=1e-3)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @nvtx.annotate("Training step", color="red", domain="my_domain")
    def training_step(self, train_batch, batch_idx):
        training_batch, training_batch_labels = train_batch['image'], train_batch['kp_label']
        x = training_batch
        training_output = self(x)
        loss = self.loss_fn(training_output, training_batch_labels)
        #self.wandb_run.log('train/loss', loss, on_step=True)
        #print("Training loss: " + str(loss.item()))        # This print statement messes ups the pytorch_lightning progress bar lol 
        self.wandb_run.log({'train/loss': loss.item()})
        #print("First outputs and labels " + str(training_output[0]) + " " + str(training_batch_labels[0].item()))
        #self.log(name="train/loss", value=loss)
        return loss

    @nvtx.annotate("Validation step", color="green", domain="my_domain")
    def validation_step(self, validation_batch, batch_idx):
        val_batch, val_batch_labels = validation_batch['image'], validation_batch['kp_label']
        full_val_batch = validation_batch['full_image']     # 'Full' here means the images without subset_pixel applied
        img_names = validation_batch['img_name']
        x = val_batch
        val_output = self(x)
        loss = self.loss_fn(val_output, val_batch_labels)
        #print("Validation loss: " + str(loss.item()))      # This print statement messes ups the pytorch_lightning progress bar lol
        self.wandb_run.log({'validation/loss': loss.item()})


        # * Logging the predictions
        # Must remember that val_output is a tensor of shape (batch_size, 2 * num_keypoints)
        # And x is a tensor of shape (batch_size, 1, self.image_height, self.image_width)
        """
        fig_output = plot_val_images(images=full_val_batch, preds=val_output, labels=val_batch_labels, img_names=img_names, num_keypoints=self.num_keypoints, title='Unsubsetted Image')
        self.wandb_run.log({f'validation/val_output_{batch_idx}': fig_output})
        # Plot the model output from what the model actually sees
        fig_subsetted_output = plot_val_images(images=val_batch, preds=val_output, labels=val_batch_labels, img_names=img_names, num_keypoints=self.num_keypoints, title='Model View')
        self.wandb_run.log({f'validation/epoch_{str(self.current_epoch)}_val_subsetted_output_{batch_idx}': fig_subsetted_output})
        # Just plot the input images
        fig_input = plot_inputs(images=full_val_batch, img_names=img_names, title='Input Image')
        self.wandb_run.log({f'validation/epoch_{str(self.current_epoch)}_val_input_{batch_idx}': fig_input})
        """

        # Use plot_test_images to plot the images
        # ! Disabling val plotting for now to see how the models do
        # Do this block if epoch is divisible by 20
        """
        if self.current_epoch % 20 == 0:
            fig_output_vector = plot_outputs(images=full_val_batch, preds=val_output, labels=val_batch_labels, img_names=img_names, num_keypoints=self.num_keypoints, title='Unsubsetted Image')
            fig_subsetted_output_vector = plot_outputs(images=val_batch, preds=val_output, labels=val_batch_labels, img_names=img_names, num_keypoints=self.num_keypoints, title='Model View')
            fig_intput_vector = plot_inputs(images=full_val_batch, img_names=img_names, title='Input Image')

            for i in range(len(fig_output_vector)):
                self.wandb_run.log({f'validation/{img_names[i]}/E{self.current_epoch}_full_output': fig_output_vector[i]})
                self.wandb_run.log({f'validation/{img_names[i]}/E{self.current_epoch}_subsetted_output': fig_subsetted_output_vector[i]})
                self.wandb_run.log({f'validation/{img_names[i]}/E{self.current_epoch}_input': fig_intput_vector[i]})
        """

        return loss



    @nvtx.annotate("Test step", color="blue", domain="my_domain")
    def test_step(self, test_batch, batch_idx):
        input_test_batch, test_batch_labels = test_batch['image'], test_batch['kp_label']          # 'input' here means the images inputted to the model, often with subset_pixel applied
        full_test_batch = test_batch['full_image']     # 'Full' here means the images without subset_pixel applied
        img_names = test_batch['img_name']
        x = input_test_batch
        test_output = self(x)
        loss = self.loss_fn(test_output, test_batch_labels)

        """
        for i in range(len(test_output)):
            # We will plot the outputs, which are 2D keypoints, to self.csv_file in the row that corresponds to the image name
            keypoints = test_output[i].detach().cpu().numpy()
            # Make the keypoints a 2D array of shape (num_keypoints, 2)
            keypoints = keypoints.reshape(self.num_keypoints, 2)
            #keypoints.view(64,2)
            img_name = img_names[i]

            # Find the row that corresponds to the image name
            #self.csv_file.loc[self.csv_file['Image address'] == img_name]['Femur PR KP points'] = str(keypoints)
            self.csv_file.loc[self.csv_file['Image address'] == img_name, 'Femur PR KP points'] = str(keypoints)
        """

        # Logging the predictions
        # Must remember that test_output is a tensor of shape (batch_size, 2 * num_keypoints)
        # And x is a tensor of shape (batch_size, 1, self.image_height, self.image_width)
        data_set_name = 'naive_set' if self.config.datamodule['USE_NAIVE_TEST_SET'] else 'test_set'
        fig_output_vector = plot_outputs(images=full_test_batch, preds=test_output, labels=test_batch_labels, img_names=img_names, num_keypoints=self.num_keypoints, title='Unsubsetted Image')
        fig_subsetted_output_vector = plot_outputs(images=input_test_batch, preds=test_output, labels=test_batch_labels, img_names=img_names, num_keypoints=self.num_keypoints, title='Model View')
        fig_input_vector = plot_inputs(images=full_test_batch, img_names=img_names, title='Input Image')

        for i in range(len(fig_output_vector)):
            self.wandb_run.log({f'test/{data_set_name}/{img_names[i]}/full_output': fig_output_vector[i]})
            self.wandb_run.log({f'test/{data_set_name}/{img_names[i]}/subsetted_output': fig_subsetted_output_vector[i]})
            self.wandb_run.log({f'test/{data_set_name}/{img_names[i]}/input': fig_input_vector[i]})



        return loss
    
