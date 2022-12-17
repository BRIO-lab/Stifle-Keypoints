# let's try summ out


import torch
import torch.nn as nn
import torchvision
import numpy as np

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

        self.my_dict = nn.ModuleDict({})

        self.my_dict["pre_block"] = nn.Sequential(
            nn.Conv2d(self.in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        
        #self.my_dict["resnet"] = make_resnet(3, 64, 3, 1)
        # this 
        self.my_dict["resnet"] = torchvision.models.resnet34(pretrained=True)

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
        self.wandb_run.log({'train/loss': loss.item()})
        #self.log(name="train/loss", value=loss)
        return loss

    @nvtx.annotate("Validation step", color="green", domain="my_domain")
    def validation_step(self, validation_batch, batch_idx):
        val_batch, val_batch_labels = validation_batch['image'], validation_batch['label']
        x = val_batch
        val_output = self(x)
        loss = self.loss_fn(val_output, val_batch_labels)
        self.wandb_run.log({'validation/loss': loss.item()})
        # TODO: Cannot log images to wandb because the images have 128(=num_keypoints) channels
        #image = wandb.Image(val_output, caption='Validation output')
        #self.wandb_run.log({'val_output': image})
        return loss

    @nvtx.annotate("Test step", color="blue", domain="my_domain")
    def test_step(self, test_batch, batch_idx):
        test_batch, test_batch_labels = test_batch['image'], test_batch['label']
        x = test_batch
        test_output = self(x)
        loss = self.loss_fn(test_output, test_batch_labels)
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
