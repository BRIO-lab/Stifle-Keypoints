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
            nn.Conv2d(self.in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        
        #self.my_dict["resnet"] = make_resnet(3, 64, 3, 1)
        # this 
        self.my_dict["resnet"] = torchvision.models.resnet34(pretrained=True)

        assert self.num_keypoints <= 50, "If num_keypoints > 50, the last linear layer is growing bigger, which seems unreasonable."

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
        training_batch, training_batch_labels = train_batch['image'], train_batch['label']
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
        val_batch, val_batch_labels = validation_batch['image'], validation_batch['label']
        img_names = validation_batch['img_name']
        x = val_batch
        val_output = self(x)
        loss = self.loss_fn(val_output, val_batch_labels)
        print("Validation loss: " + str(loss.item()))
        self.wandb_run.log({'validation/loss': loss.item()})


        # Logging the predictions
        # Must remember that val_output is a tensor of shape (batch_size, 2 * num_keypoints)
        # And x is a tensor of shape (batch_size, 1, self.image_height, self.image_width)
        num_images = val_batch.shape[0]
        output = val_output.view(num_images, self.num_keypoints, 2)
        fig, ax = matplotlib.pyplot.subplots(1, num_images, figsize=(10, 10), squeeze=False)
        # Move everything to CPU
        val_batch = val_batch.cpu()
        output = output.cpu()
        # Flatten ax so it doesn't whine and moan
        ax = ax.flatten()
        for i in range(0, num_images):
            output[i][:, 0] = +1 * output[i][:, 0] * 512 - 0 + 128
            output[i][:, 1] = -1 * output[i][:, 1] * 512 + 512 + 256
            # Do some stuff so that img is shown correctly
            img = val_batch[i].numpy()
            img = np.transpose(img, (1, 2, 0))
            img = np.dstack((img, img, img))
            ax[i].imshow((img * 255).astype(np.uint8))  # The multiplying by 255 and stuff is so it doesn't get clipped or something
            for j in range(self.num_keypoints):
                ax[i].text(output[i][j, 0], output[i][j, 1], str(j), color='red')
            image_name = img_names[i].split('/')[-1]    # Format img_names[i] so that only the part after the last '/' is shown
            ax[i].set_title('Image {}'.format(image_name))
        self.wandb_run.log({'validation/val_output': fig})


        return loss



    @nvtx.annotate("Test step", color="blue", domain="my_domain")
    def test_step(self, test_batch, batch_idx):
        test_batch_imgs, test_batch_labels = test_batch['image'], test_batch['label']
        img_names = test_batch['img_name']
        x = test_batch_imgs
        test_output = self(x)
        loss = self.loss_fn(test_output, test_batch_labels)


        # Logging the predictions
        num_images = test_batch_imgs.shape[0]
        output = test_output.view(num_images, self.num_keypoints, 2)
        fig, ax = matplotlib.pyplot.subplots(1, num_images, figsize=(10, 10), squeeze=False)
        # Move everything to CPU
        test_batch_imgs = test_batch_imgs.cpu()
        output = output.cpu()
        output = np.array(output, dtype=np.float64)
        #output = +1 * output * 512 + 512
        # Flatten ax so it doesn't whine and moan
        ax = ax.flatten()
        for i in range(0, num_images):
            output[i][:, 0] = +1 * output[i][:, 0] * 512 - 0 + 128
            output[i][:, 1] = -1 * output[i][:, 1] * 512 + 512 + 256
            # Do some stuff so that img is shown correctly
            img = test_batch_imgs[i].numpy()
            img = np.transpose(img, (1, 2, 0))
            # Transpose the output so that it's the same way as img
            #output[i] = np.transpose(output[i], (1, 0))
            img = np.dstack((img, img, img))
            ax[i].imshow((img * 255).astype(np.uint8))
            for j in range(self.num_keypoints):
                #ax[i].text(output[i][j, 0], output[i][j, 1], str(j), color='red')
                ax[i].text(output[i][j, 0], output[i][j, 1], str(j), color='red')
                #print("type of output[i][j, 0].item(): " + str(type(output[i][j, 0].item())))
                #ax[i].text(output[i][j, 0] * 512 + 512, output[i][j, 1] * 512 + 512, str(j), color='red')
                print(f'point {j} is ' + str(output[i][j, 0]) + " " + str(output[i][j, 1]))
            image_name = img_names[i].split('/')[-1]    # Format img_names[i] so that only the part after the last '/' is shown
            ax[i].set_title('Image {}'.format(image_name))
        self.wandb_run.log({f'test/output_batch_{batch_idx}': fig})



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
