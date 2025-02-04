import os
import sys
import time
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader, IterableDataset, random_split
from PIL import Image
from tensorboardX import SummaryWriter
import random
from datasets import load_dataset
from options.train_options import TrainOptions
from tqdm import tqdm
import itertools
import wandb
import gc
import psutil

import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
from huggingface_hub import login

from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from data import create_dataloader, data_augment, custom_resize, LocalDataset, LimitedDataset, CombinedLimitedDataset
from earlystop import EarlyStopping
from networks.trainer import Trainer
from accelerate import Accelerator
from validate import validate

from util import flush, print_memory_usage

ImageFile.LOAD_TRUNCATED_IMAGES = True
accelerator = Accelerator(log_with="wandb")

if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.name = "diffusion_blur_jpg_prob0.5"
    opt.blur_prob = 0.5 
    opt.blur_sig = [0.0, 3.0] 
    opt.jpg_prob = 0.5 
    opt.jpg_method = ['cv2','pil']
    opt.jpg_qual = [30,100]
    
    if accelerator.is_main_process:
        accelerator.init_trackers("cnndetector", config=vars(opt))
    
    # Load local dataset
    local_data_path = opt.dataroot
    crop_func = transforms.RandomCrop(opt.cropSize)
    flip_func = transforms.RandomHorizontalFlip()
    rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))

    transform = transforms.Compose([
                    rz_func,
                    transforms.Lambda(lambda img: data_augment(img, opt)),
                    crop_func,
                    flip_func,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    local_dataset = LocalDataset(local_data_path)
    
    train_split_ratio = 0.995
    train_size = int(train_split_ratio * len(local_dataset))
    val_size = len(local_dataset) - train_size
    local_train_dataset, local_val_dataset = random_split(local_dataset, [train_size, val_size])

    # Load Huggingface dataset
    huggingface_dataset_name = 'imagenet-1k'
    huggingface_dataset = load_dataset(huggingface_dataset_name, split='test', use_auth_token=True, streaming=True)
    
    # Create DataLoader
    batch_size = opt.batch_size
    
    val_dataset = load_dataset(huggingface_dataset_name, split='validation', use_auth_token=True, streaming=True)
    limited_dataset = CombinedLimitedDataset(local_val_dataset, val_dataset, max_size=1000, transform=transform, dataset_1_is_local=True)
    data_loader_val = DataLoader(limited_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    
    combined_dataset = CombinedLimitedDataset(local_dataset, huggingface_dataset, max_size=200000, transform=transform, dataset_1_is_local=True)
    data_loader = DataLoader(combined_dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True)

    model = Trainer(opt)
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    
    data_loader, data_loader_val = accelerator.prepare(
        data_loader, data_loader_val
    )

    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        with tqdm(data_loader) as t:
            for (combined_data, combined_labels) in t:
                model.total_steps += 1
                epoch_iter += opt.batch_size

                model.set_input((combined_data, combined_labels))
                model.optimize_parameters()

                if model.total_steps % opt.loss_freq == 0:
                    print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                    accelerator.log({"loss": model.loss})

                if model.total_steps % opt.save_latest_freq == 0:
                    print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                          (opt.name, epoch, model.total_steps))
                    model.save_networks('latest')

                if (model.total_steps % opt.validation_frequency == 0):
                    model.eval()
                    acc, ap, r_acc, f_acc = validate(model.model, opt, data_loader_val)[:4]
                    accelerator.log({"accuracy": acc, "ap": ap, "r_acc": r_acc, "f_acc": f_acc})
                    if accelerator.is_main_process:
                        print("(Val @ step {}) acc: {}; ap: {}; r_acc: {}; f_acc: {}".format(model.total_steps, acc, ap, r_acc, f_acc))
                    model.train()
                
                flush()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)
            
        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break
        model.train()
