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
from data import create_dataloader, data_augment, custom_resize, ImageDataset, transform_func
from earlystop import EarlyStopping
from networks.trainer_vit import Trainer
from accelerate import Accelerator
from validate import validate

from util import flush, print_memory_usage
from pathlib import Path
import random


ImageFile.LOAD_TRUNCATED_IMAGES = True
accelerator = Accelerator(log_with="wandb")

train_split_ratio = 0.995

class PTFileDataset(Dataset):
    def __init__(self, root_dir, num_indexes=6):
        self.root_dir = Path(root_dir)
        self.pt_files = list(self.root_dir.glob('**/*.pt'))
        self.labels = [str(file.parent.name) for file in self.pt_files]

        # create a mapping from label to integer
        self.label_to_int = {
            '0_real': 0,
            '1_fake': 1
        }

        # get the unique base paths (without the index) of the pt files
        base_files = list(set([str(file).rsplit('_', 1)[0] for file in self.pt_files]))

        # precompute the groups of augmented files
        self.augmented_files = []
        for base_file in base_files:
            aug_files = [Path(f"{base_file}_{i}.pt") for i in range(num_indexes)]
            self.augmented_files.append(aug_files)

    def __getitem__(self, idx):
        aug_files = self.augmented_files[idx]
        pt_file = random.choice(aug_files)
        data = torch.load(pt_file)
        label = self.label_to_int[pt_file.parent.name]
        return data, label

    def __len__(self):
        return len(self.augmented_files)
    


def get_data_loaders(opt):
    batch_size = opt.batch_size
    # local_dataset = PTFileDataset(opt.dataroot, num_indexes=1)
    local_dataset = ImageDataset(opt.dataroot, transform=transform_func(opt))
    
    train_size = int(train_split_ratio * len(local_dataset))
    val_size = len(local_dataset) - train_size
    train_dataset, val_dataset = random_split(local_dataset, [train_size, val_size])

    data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True)
    data_loader_val = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    
    return data_loader, data_loader_val

if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.blur_prob = 0.5 
    opt.blur_sig = [0.0, 3.0] 
    opt.jpg_prob = 0.5 
    opt.jpg_method = ['cv2','pil']
    opt.jpg_qual = [30,100]
    
    if accelerator.is_main_process:
        accelerator.init_trackers("cnndetector", config=vars(opt))
    
    data_loader, data_loader_val = get_data_loaders(opt)

    model = Trainer(opt)
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    
    data_loader, data_loader_val = accelerator.prepare(
        data_loader, data_loader_val
    )

    for epoch in range(opt.niter):
        epoch_iter = 0

        with tqdm(data_loader) as t:
            for (data, labels) in t:
                model.total_steps += 1
                epoch_iter += opt.batch_size

                model.set_input((data, labels))
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
