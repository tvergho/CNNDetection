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
from data import create_dataloader, data_augment, custom_resize
from earlystop import EarlyStopping
from networks.trainer_new import Trainer
from accelerate import Accelerator
from validate import validate

from util import flush, print_memory_usage
from pathlib import Path
import random

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
train_split_ratio = 0.95

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
    local_dataset = PTFileDataset(opt.dataroot, num_indexes=1)
    
    train_size = int(train_split_ratio * len(local_dataset))
    val_size = len(local_dataset) - train_size
    train_dataset, val_dataset = random_split(local_dataset, [train_size, val_size])

    data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True)
    data_loader_val = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    
    return data_loader, data_loader_val

if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader, data_loader_val = get_data_loaders(opt)

    X_train = []
    y_train = []
    for data, labels in data_loader:
        X_train.append(data.detach().numpy())
        y_train.append(labels.detach().numpy())

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    # Flatten the data
    X_train = X_train.reshape(X_train.shape[0], -1)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Train the SVM
    clf = svm.SVC()
    clf.fit(X_train, y_train)
