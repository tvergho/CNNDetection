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
from data import data_augment
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

crop_size = 256
opt = TrainOptions().parse()

unet = UNet2DModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="unet").cuda()
vqvae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae").cuda()
scheduler = DDIMScheduler.from_pretrained("CompVis/ldm-celebahq-256", subfolder="scheduler")

unet.enable_xformers_memory_efficient_attention()
vqvae.enable_xformers_memory_efficient_attention()

@torch.compile
def get_image_dire(image):
    with autocast("cuda"), torch.no_grad():
        latents = vqvae.encode(image).latents

        scheduler.set_timesteps(20)
        for i, e in enumerate(np.flip(scheduler.timesteps, 0)):
            latents = scheduler.reverse_step(unet(latents, e).sample, e, latents).next_sample

        latents_ = latents.clone()

        for i, e in enumerate(scheduler.timesteps):
            latents_ = scheduler.step(unet(latents_, e).sample, e, latents_).prev_sample

        dire = calculate_dire(image, latents_)
        return dire

@torch.compile
def calculate_dire(image, latents_):
    # Decode the latent vectors
    decoded_image1 = image
    decoded_image2 = vqvae.decode(latents_).sample

    # Ensure the images are in the same range (0, 1)
    decoded_image1 = (decoded_image1 / 2 + 0.5).clamp(0, 1)
    decoded_image2 = (decoded_image2 / 2 + 0.5).clamp(0, 1)

    # Compute the DIRE (the absolute difference between the images)
    dire = torch.abs(decoded_image1 - decoded_image2)

    return dire


transform_func = transforms.Compose([
                transforms.Lambda(lambda img: data_augment(img, opt)),
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

def transform(img):
    return [transform_func(img)]

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
    
def augmented_process_and_save_images(input_dir, output_dir, batch_size):
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".png"):  # or whatever your image file extension is
                image_paths.append(Path(root) / file)

    dataset = ImageDataset(image_paths, transform, output_dir, input_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    for augmented_images_batch, batch_image_paths in tqdm(dataloader):
        # Process each batch of augmented images
        for i, input_images in enumerate(augmented_images_batch):
            input_images = input_images.to('cuda')  # Move to GPU

            # Run the model and save the output feature vectors
            feature_vectors = get_image_dire(input_images)
            
            for j, img in enumerate(feature_vectors):
                image_path = batch_image_paths[j]
                output_image_path = output_dir / Path(image_path).relative_to(input_dir)
                
                # Append the index of the augmentation to the filename
                output_image_path = output_image_path.with_stem(f"{output_image_path.stem}")
                output_image_path = output_image_path.with_suffix('.png')

                # Skip if vector already exists
                if output_image_path.exists():
                    continue

                # Ensure the output subdirectory exists
                output_image_path.parent.mkdir(parents=True, exist_ok=True)
                
                dire_image = transforms.ToPILImage()(img.cpu().detach())
                dire_image.save(str(output_image_path))
                # Save the feature vector
                # torch.save(feature_vector.cpu(), str(output_image_path))  # Move to CPU before saving

augmented_process_and_save_images(Path("dataset/train"), Path("dataset/dire"), batch_size=16)