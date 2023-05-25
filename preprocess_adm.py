from options.train_options import TrainOptions
from networks.trainer import Trainer
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision.transforms.functional as TF
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from data import data_augment, numpy_to_pil_image, LocalDataset
import os
from pathlib import Path
import os
from pathlib import Path
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from torchvision.transforms import ToPILImage
from networks.ef import FeatureExtractionEfficientNet
from diffusers import LDMPipeline, UNet2DModel, VQModel, DDIMScheduler
from torch import Tensor, autocast, inference_mode
import numpy as np
from datasets import load_dataset
import io
from adm import DiffusionModel

batch_size = 2
model = DiffusionModel(batch_size=batch_size)
# model = DistributedDataParallel(model)

crop_size = 256
opt = TrainOptions().parse()

if not dist.is_initialized():
    dist.init_process_group(backend='nccl', init_method='env://')
torch.cuda.set_device(opt.local_rank)

transform_func = transforms.Compose([
                transforms.Lambda(lambda img: data_augment(img, opt)),
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

def transform(img):
    return [transform_func(img)]

class StreamingImageDataset(Dataset):
    def __init__(self, images, transform, output_dir, label, is_huggingface=False):
        self.transform = transform
        self.label = label
        self.images = iter(images) if is_huggingface else images
        self.output_dir = output_dir
        self.is_huggingface = is_huggingface
    
    def label_to_string(self):
        return "real" if self.label == 0 else "fake"
    
    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        image = next(self.images)['image'] if self.is_huggingface else self.images[idx]
        
        
        image = numpy_to_pil_image(image if self.is_huggingface else image[0])
        image = self.transform(image)
        # image = image.unsqueeze(0)  # Add an extra dimension for the batch
        return image, idx

def augmented_process_and_save_images(dataset, batch_size, label):
    # Ensure output directory exists
    Path(dataset.output_dir / f"{label}_{dataset.label_to_string()}").mkdir(parents=True, exist_ok=True)

    image_counter = 0
    
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, sampler=sampler)
    
    for images, indexes in tqdm(dataloader):
        images = images.to('cuda')
        
        # Run the model and save the output feature vectors
        feature_vectors = model.get_image_dire(images)
            
        for i, img in enumerate(feature_vectors):
            output_image_path = dataset.output_dir / f"{label}_{dataset.label_to_string()}" / f"{indexes[i]}.png"
            image_counter += 1

            # Skip if vector already exists
            if output_image_path.exists():
                continue

            # Ensure the output subdirectory exists
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            
            dire_image = transforms.ToPILImage()(img.cpu().detach())
            dire_image.save(str(output_image_path))

# Preprocess real images
real_dataset_name = 'imagenet-1k'
real_dataset = load_dataset(real_dataset_name, split='test', use_auth_token=True, streaming=True)
real_image_dataset = StreamingImageDataset(real_dataset, transform_func, Path("dataset/dire"), label=0)
augmented_process_and_save_images(real_image_dataset, batch_size=8, label=0)

# Preprocess fake images
# local_data_path = Path("dataset/fake/imagenet1k")
# fake_dataset = LocalDataset(local_data_path)
# fake_image_dataset = StreamingImageDataset(fake_dataset, transform_func, Path("dataset/direadm"), label=1)
# augmented_process_and_save_images(fake_image_dataset, batch_size=batch_size, label=1)