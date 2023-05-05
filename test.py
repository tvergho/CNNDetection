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
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from accelerate import Accelerator

ImageFile.LOAD_TRUNCATED_IMAGES = True
accelerator = Accelerator(log_with="wandb")

def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])


class LocalDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        image_paths = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        return img, 0  # "fake" class label
    
def validate(model, opt, data_loader):
    with torch.no_grad():
        y_true, y_pred = [], []
        with tqdm(data_loader) as t:
            for img, label in t:
                in_tens = img.cuda()
                y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred

            
def numpy_to_pil_image(img):
    img = np.asarray(img)        
    if len(img.shape) == 2:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        
    # img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


class CombinedDataset(Dataset):
    def __init__(self, local_dataset, huggingface_dataset, transform=None):
        self.local_dataset = local_dataset
        self.huggingface_dataset = huggingface_dataset
        self.transform = transform
        self.huggingface_iter = iter(self.huggingface_dataset)

    def __len__(self):
        return 200000

    def __getitem__(self, idx):
        if idx < len(self.local_dataset):
            img, label = self.local_dataset[idx]
        else:
            try:
                img = next(self.huggingface_iter)['image']
            except StopIteration:
                # Reset the iterator when the end is reached
                self.huggingface_iter = iter(self.huggingface_dataset)
                img = next(self.huggingface_iter)['image']
            label = 1

        img = numpy_to_pil_image(img)

        if self.transform:
            img = self.transform(img)

        return img, label

    
class LimitedDataset(Dataset):
    def __init__(self, dataset, max_size):
        self.dataset = dataset
        self.max_size = min(max_size, len(dataset))

    def __len__(self):
        return self.max_size

    def __getitem__(self, idx):
        return self.dataset[idx]

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    # print(f'Memory allocated: {torch.cuda.memory_allocated()}, Memory cached: {torch.cuda.memory_cached()}')

def print_memory_usage():
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total / (1024 ** 2)  # Convert to MB
    used_memory = memory_info.used / (1024 ** 2)  # Convert to MB
    available_memory = memory_info.available / (1024 ** 2)  # Convert to MB
    percent_used = memory_info.percent

    print(f"Total memory: {total_memory:.2f} MB")
    print(f"Used memory: {used_memory:.2f} MB")
    print(f"Available memory: {available_memory:.2f} MB")
    print(f"Percent used: {percent_used}%")

if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.name = "diffusion_blur_jpg_prob0.5"
    opt.blur_prob = 0.5 
    opt.blur_sig = [0.0, 3.0] 
    opt.jpg_prob = 0.5 
    opt.jpg_method = ['cv2','pil']
    opt.jpg_qual = [30,100]
    opt.validation_frequency = 200
    
    if accelerator.is_main_process:
        accelerator.init_trackers("cnndetector", config=vars(opt))
    
    # Load local dataset
    local_data_path = 'imagenet1k'
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
    combined_val_dataset = CombinedDataset(local_val_dataset, val_dataset, transform)
    limited_dataset = LimitedDataset(combined_val_dataset, max_size=1000)
    data_loader_val = DataLoader(limited_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    
    combined_dataset = CombinedDataset(local_dataset, huggingface_dataset, transform)
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
                    # train_writer.add_scalar('loss', model.loss, model.total_steps)
                    accelerator.log({"loss": model.loss})

                if model.total_steps % opt.save_latest_freq == 0:
                    print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                          (opt.name, epoch, model.total_steps))
                    model.save_networks('latest')

                if (model.total_steps % opt.validation_frequency == 0):
                    model.eval()
                    acc, ap, r_acc, f_acc = validate(model.model, opt, data_loader_val)[:4]
                    # val_writer.add_scalar('accuracy', acc, model.total_steps)
                    # val_writer.add_scalar('ap', ap, model.total_steps)
                    accelerator.log({"accuracy": acc, "ap": ap, "r_acc": r_acc, "f_acc": f_acc})
                    if accelerator.is_main_process:
                        print("(Val @ step {}) acc: {}; ap: {}; r_acc: {}; f_acc: {}".format(model.total_steps, acc, ap, r_acc, f_acc))
                    model.train()
                
                flush()
                # print_memory_usage()

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
