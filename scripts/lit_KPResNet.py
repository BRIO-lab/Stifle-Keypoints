# let's try summ out


import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
            gaussian_sigma=self.config.dataset['GAUSSIAN_STDDEV']
            )

        
        self.in_channels = self.config.dataset['IMG_CHANNELS']
        self.num_keypoints = self.config.dataset['NUM_KEY_POINTS']
        self.batch_size = self.config.datamodule['BATCH_SIZE']      # This is used in the logging steps in validation_step and test_step.
        self.image_height = self.config.dataset['IMAGE_HEIGHT']     # These two, too.
        self.image_width = self.config.dataset['IMAGE_WIDTH']

        self.my_dict = nn.ModuleDict({})

        self.my_dict["pre_block"] = nn.Sequential(
            nn.Conv2d(self.in_channels, 3, kernel_size=1, stride=1, padding=0, bias=False),     # This is just to convert the input channels to 3, which is what the resnet expects.
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        
        #self.my_dict["resnet"] = make_resnet(3, 64, 3, 1)
        # this 
        self.my_dict["resnet"] = torchvision.models.resnet34(weights='IMAGENET1K_V1')

        #assert self.num_keypoints <= 50, "If num_keypoints > 50, the last linear layer is growing bigger, which seems unreasonable."
        # The above assertion does not need to be made because it could be that the model is just finding a lower-dimensional representation of the keypoints, which, if accurate, would be a good thing.
        # ! If model performance is bad is bad when using 64 keypoints, then it may be a good idea to reexamine this.

        self.my_dict["keypoints"] = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 2 * self.num_keypoints)
        )

    def forward(self, x):
        """This performs a forward pass on the dataset.

        Args:
            x (torch.Tensor): This is a tensor containing the input data.

        Returns:
            the forward pass of the dataset.
        """
        x = self.my_dict["pre_block"](x)
        x = self.my_dict["resnet"](x)
        x = self.my_dict["keypoints"](x)
        return x

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
        print("Training loss: " + str(loss.item()))
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
        print("Validation loss: " + str(loss.item()))
        self.wandb_run.log({'validation/loss': loss.item()})


        # * Logging the predictions
        # Must remember that val_output is a tensor of shape (batch_size, 2 * num_keypoints)
        # And x is a tensor of shape (batch_size, 1, self.image_height, self.image_width)
        fig_output = plot_val_images(images=full_val_batch, preds=val_output, labels=val_batch_labels, img_names=img_names, num_keypoints=self.num_keypoints, title='Unsubsetted Image')
        self.wandb_run.log({f'validation/val_output_{batch_idx}': fig_output})
        # Plot the model output from what the model actually sees
        fig_subsetted_output = plot_val_images(images=val_batch, preds=val_output, labels=val_batch_labels, img_names=img_names, num_keypoints=self.num_keypoints, title='Model View')
        self.wandb_run.log({f'validation/epoch_{str(self.current_epoch)}_val_subsetted_output_{batch_idx}': fig_subsetted_output})
        # Just plot the input images
        fig_input = plot_inputs(images=full_val_batch, img_names=img_names, title='Input Image')
        self.wandb_run.log({f'validation/epoch_{str(self.current_epoch)}_val_input_{batch_idx}': fig_input})


        return loss



    @nvtx.annotate("Test step", color="blue", domain="my_domain")
    def test_step(self, test_batch, batch_idx):
        test_batch_imgs, test_batch_labels = test_batch['image'], test_batch['label']
        plotting_imgs= test_batch['raw_image'] if self.config.dataset['SUBSET_PIXELS'] else test_batch['image']       # Using the raw image instead of the subsetted one
        img_names = test_batch['img_name']
        x = test_batch_imgs
        test_output = self(x)
        loss = self.loss_fn(test_output, test_batch_labels)


        # Logging the predictions
        # TODO: Maybe this should be moved to a separate helper function in utility.py?
        num_images = test_batch_imgs.shape[0]
        output = test_output.view(num_images, self.num_keypoints, 2)
        fig, ax = matplotlib.pyplot.subplots(1, num_images, figsize=(10, 10), squeeze=False)
        # Move everything to CPU
        plotting_imgs = plotting_imgs.cpu()
        output = output.cpu()
        output = np.array(output, dtype=np.float64)
        labels = test_batch_labels.cpu()
        labels = labels.numpy()
        # Flatten ax so it doesn't whine and moan
        ax = ax.flatten()
        for i in range(0, num_images):
            output[i][:, 0] = +1 * output[i][:, 0] * 1024
            output[i][:, 1] = -1 * output[i][:, 1] * 1024 + 1024
            labels[i][:, 0] = +1 * labels[i][:, 0] * 1024
            labels[i][:, 1] = -1 * labels[i][:, 1] * 1024 + 1024
            # Do some stuff so that img is shown correctly
            img = plotting_imgs[i].numpy()
            img = np.transpose(img, (1, 2, 0))  # Transpose the output so that it's the same way as img
            img = np.dstack((img, img, img))    # Make it 3 channels
            ax[i].imshow((img * 255).astype(np.uint8))
            for j in range(self.num_keypoints):
                ax[i].text(labels[i][j, 0], labels[i][j, 1], str(j), color='blue')
                ax[i].plot(labels[i][j, 0], labels[i][j, 1], 'g.')
                ax[i].plot(output[i][j, 0], output[i][j, 1], 'r.')
            image_name = img_names[i].split('/')[-1]    # Format img_names[i] so that only the part after the last '/' is shown
            ax[i].set_title('Image {}'.format(image_name))
        self.wandb_run.log({f'test/test_batch_{batch_idx}': fig})



        return loss
    




    #def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx) -> None:
    """
    def on_test_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx, **kwargs) -> None:
        print(outputs.size())
        for image in outputs:
            image = self.wandb.Image(image, caption='Test output from batch ' + str(batch_idx))
            self.wandb_run.log({'test_output': image})
        #return super().on_test_batch_end(trainer, pl_module, batch, batch_idx)
        #return super().on_test_batch_end(batch, batch_idx)
    """

"""
    def train_dataloader(self):
        return

    def val_dataloader(self):
        return
"""

    # def backward():
    # def optimizer_step():
