from options.train_options import TrainOptions
# from networks.trainer import Trainer
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
# from networks.ef import FeatureExtractionEfficientNet
from diffusers import LDMPipeline, UNet2DModel, VQModel, DDIMScheduler
from torch import Tensor, autocast, inference_mode
import numpy as np
from datasets import load_dataset
import io
from adm import DiffusionModel
import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method
import concurrent.futures

opt = TrainOptions().parse()

if not dist.is_initialized():
    dist.init_process_group(backend='nccl', init_method='env://')

torch.cuda.set_device(opt.local_rank)

batch_size = 4
model = DiffusionModel(batch_size=batch_size)

crop_size = 256

transform_func = transforms.Compose([
                # transforms.Lambda(lambda img: data_augment(img, opt)),
                transforms.RandomResizedCrop(crop_size),
                # transforms.RandomHorizontalFlip(),
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

def augmented_process_and_save_images(dataset, batch_size, label, gpu_id):
    # Ensure output directory exists
    Path(dataset.output_dir / f"{label}_{dataset.label_to_string()}").mkdir(parents=True, exist_ok=True)

    image_counter = 0
    
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, sampler=sampler)
    
    for images, indexes in tqdm(dataloader):
        images = images.to('cuda')
        
        found = False
        for index in indexes:
            output_image_path = dataset.output_dir / f"{label}_{dataset.label_to_string()}" / f"{index}.png"
            
            if output_image_path.exists():
                found = True
        
        if found:
            continue
            
        # Run the model and save the output feature vectors
        feature_vectors = model.get_image_dire(images)
            
        for i, img in enumerate(feature_vectors):
            # img, recon, og = v
            
            output_image_path = dataset.output_dir / f"{label}_{dataset.label_to_string()}" / f"{indexes[i]}.png"
            image_counter += 1

            # Skip if vector already exists
            # if output_image_path.exists():
            #     continue

            # Ensure the output subdirectory exists
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            
            dire_image = transforms.ToPILImage()(img.cpu().detach())
            dire_image.save(str(output_image_path))

def augmented_process_and_save_images_all(dataset, batch_size, label, gpu_id):
    # Ensure output directory exists
    output_dir = dataset.output_dir / f"{label}_{dataset.label_to_string()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_counter = 0
    
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, sampler=sampler)
    
    for images, indexes in tqdm(dataloader):
        images = images.to('cuda')
        
        found = False
        for index in indexes:
            if any((output_dir / f"{index}_{img_type}.png").exists() for img_type in ["original", "reconstructed", "dire"]):
                found = True
        
        if found:
            continue
            
        # Run the model and save the output feature vectors
        original_img, reconstructed_img, dire_img = model.get_image_dire(images)
        
        for i, v in enumerate(original_img):
            for img, img_type in zip([original_img[i], reconstructed_img[i], dire_img[i]], ["original", "reconstructed", "dire"]):
                output_image_path = output_dir / f"{indexes[i]}_{img_type}.png"
                image_counter += 1

                # Ensure the output subdirectory exists
                output_image_path.parent.mkdir(parents=True, exist_ok=True)

                # Save the image
                img_pil = transforms.ToPILImage()(img.cpu().detach())
                img_pil.save(str(output_image_path))

# New function for processing the fake dataset
def process_fake_dataset(gpu_id):
    torch.cuda.set_device(gpu_id)
    fake_data_path = "dataset/fake/imagenet1k"
    fake_dataset = LocalDataset(fake_data_path)
    fake_image_dataset = StreamingImageDataset(fake_dataset, transform_func, Path("dataset/direadm4"), label=1)
    augmented_process_and_save_images(fake_image_dataset, batch_size, label=1, gpu_id=gpu_id)

# New function for processing the real dataset
def process_real_dataset(gpu_id=0):
    # torch.cuda.set_device(gpu_id)
    real_dataset_name = "imagenet-1k"
    real_dataset = load_dataset(real_dataset_name, split="test", use_auth_token=True, streaming=True)
    real_image_dataset = StreamingImageDataset(real_dataset, transform_func, Path("dataset/direadmsd-diffusiondb"), label=0, is_huggingface=True)
    augmented_process_and_save_images(real_image_dataset, batch_size, label=0, gpu_id=gpu_id)
    
def process_diffusion_dataset(gpu_id=0):
    # torch.cuda.set_device(gpu_id)
    real_dataset_name = "poloclub/diffusiondb"
    real_dataset = load_dataset(real_dataset_name, '2m_random_10k', split="train", use_auth_token=True, streaming=True)
    real_image_dataset = StreamingImageDataset(real_dataset, transform_func, Path("dataset/direadmadm"), label=1, is_huggingface=True)
    augmented_process_and_save_images(real_image_dataset, batch_size, label=1, gpu_id=gpu_id)

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform, output_dir, input_dir):
        self.transform = transform
        self.image_paths = []
        for image_path in image_paths:
            output_image_path = output_dir / image_path.relative_to(input_dir)
            output_image_path = output_image_path.with_stem(f"{output_image_path.stem}").with_suffix('.png') # replace .png with .pt
            if not output_image_path.exists():
                self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, str(image_path)
    
def batch_process_and_save_images(input_dir, output_dir, batch_size):
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):  # or whatever your image file extension is
                image_paths.append(Path(root) / file)

    dataset = ImageDataset(image_paths, transform, output_dir, input_dir)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, sampler=sampler)

    for augmented_images_batch, batch_image_paths in tqdm(dataloader):
        for i, input_images in enumerate(augmented_images_batch):
            input_images = input_images.to('cuda')  # Move to GPU

            # Run the model and save the output feature vectors
            feature_vectors = model.get_image_dire(input_images)
            
            for j, img in enumerate(feature_vectors):
                image_path = batch_image_paths[j]
                output_image_path = output_dir / Path(image_path).relative_to(input_dir)
                
                # Append the index of the augmentation to the filename
                output_image_path = output_image_path.with_stem(f"{output_image_path.stem}")
                output_image_path = output_image_path.with_suffix('.png')

                # Ensure the output subdirectory exists
                output_image_path.parent.mkdir(parents=True, exist_ok=True)
                
                dire_image = transforms.ToPILImage()(img.cpu().detach())
                dire_image.save(str(output_image_path))


    
# Main function to manage multiprocessing
def main():
    # batch_process_and_save_images(Path("dataset/test"), Path("dataset/diretestadm"), batch_size=batch_size)
    process_real_dataset()
    # process_real_dataset(0)
    
#     with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
#         Process the fake dataset on GPUs 0 and 1
#         gpu_ids_fake = [0, 1]
#         processes_fake = [executor.submit(process_fake_dataset, gpu_id) for gpu_id in gpu_ids_fake]

#         # Process the real dataset on GPUs 2 and 3
#         gpu_ids_real = [2, 3]
#         processes_real = [executor.submit(process_real_dataset, gpu_id) for gpu_id in gpu_ids_real]

#         # Wait for all processes to finish
#         for future in concurrent.futures.as_completed(processes_fake + processes_real):
#             future.result()


if __name__ == "__main__":
    main()