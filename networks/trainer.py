import functools
import torch
import torch.nn as nn
from networks.resnet import resnet50
from networks.base_model import BaseModel, init_weights
from accelerate import Accelerator
import torchvision.models as models

from diffusers import LDMPipeline
from typing import Optional, Union, List, Tuple
from torch import Tensor, autocast, inference_mode
from tqdm import tqdm
import numpy as np

accelerator = Accelerator(mixed_precision="fp16")

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = models.efficientnet_v2_s()
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 1)
            
        if not self.isTrain or opt.continue_train:
            self.model = resnet50(num_classes=1)
            
        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)

        self.pipeline = LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256", device_map="auto")
        
        self.model, self.optimizer, self.pipeline = accelerator.prepare(
            self.model, self.optimizer, self.pipeline
        )
        
    def get_image_dire(self, image):
        latents = self.pipeline.vqvae.encode(image).latents

        self.pipeline.scheduler.set_timesteps(20)
        with autocast("cuda"), inference_mode():
            for i, e in enumerate(np.flip(self.pipeline.scheduler.timesteps, 0)):
                latents = self.pipeline.scheduler.reverse_step(self.pipeline.unet(latents, e).sample, e, latents).next_sample
                
        latents_ = latents.clone()
        
        with autocast("cuda"), inference_mode():
            for i, e in enumerate(self.pipeline.scheduler.timesteps):
                latents_ = self.pipeline.scheduler.step(self.pipeline.unet(latents_, e).sample, e, latents_).prev_sample
                
        dire = self.calculate_dire(image, latents_)
        return dire
    
    def calculate_dire(self, image, latents_):
        # Decode the latent vectors
        decoded_image1 = image
        decoded_image2 = self.pipeline.vqvae.decode(latents_).sample

        # Ensure the images are in the same range (0, 1)
        decoded_image1 = (decoded_image1 / 2 + 0.5).clamp(0, 1)
        decoded_image2 = (decoded_image2 / 2 + 0.5).clamp(0, 1)

        # Compute the DIRE (the absolute difference between the images)
        dire = torch.abs(decoded_image1 - decoded_image2)

        return dire

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        # self.input = input[0].cuda()
        # self.label = input[1].cuda().float()
        self.input = accelerator.gather(input[0])
        self.label = accelerator.gather(input[1]).float()

    def forward(self):
        dire = self.get_image_dire(self.input)
        self.output = self.model(dire)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        with autocast("cuda"):
            self.forward()
            self.loss = self.loss_fn(self.output.squeeze(1), self.label)
            self.optimizer.zero_grad()
        # self.loss.backward()
        accelerator.backward(self.loss)
        self.optimizer.step()