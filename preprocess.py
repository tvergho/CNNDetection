from options.train_options import TrainOptions
from networks.trainer import Trainer
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.model import MBConvBlock
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

crop_size = 480
opt = TrainOptions().parse()


if not dist.is_initialized():
    dist.init_process_group(backend='nccl', init_method='env://')

torch.cuda.set_device(opt.local_rank)
model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
# model.avgpool = torch.nn.Identity()
model.classifier[0] = torch.nn.Identity()
model.classifier[1] = torch.nn.Identity()
model = model.to(opt.local_rank)
model = DistributedDataParallel(model, device_ids=[opt.local_rank])

class Augmentations:
    def __init__(self, size, augment_fn, mean, std):
        self.size = size
        self.augment_fn = augment_fn
        # self.augment_fn = transforms.Lambda(lambda img: img)
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, img):
        augmented_images = []
        augments = [self.augment_fn(img, opt, noRandom=True, blur=(True if i == 1 or i == 2 else False), jpg=(True if i == 2 or i == 3 else False)) for i in range(3)]
        
        # augments = [self.augment_fn(img) for opt in range(3)]
        flips = [TF.hflip(augment) for augment in augments]
        
        for image in augments + flips:
            resized = transforms.Resize(self.size)(image)
            cropped = transforms.CenterCrop(self.size)(resized)
            to_tensor = transforms.ToTensor()(cropped)
            normalized = self.normalize(to_tensor)
            augmented_images.append(normalized)
        return augmented_images

# Usage:
transform = Augmentations(
    size=crop_size, 
    augment_fn=data_augment, 
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
)


# transform = transforms.Compose([
#                 transforms.Lambda(lambda img: data_augment(img, opt)),
#                 transforms.RandomResizedCrop(crop_size),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ])

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform, output_dir, input_dir):
        self.transform = transform
        self.image_paths = []
        for image_path in image_paths:
            output_image_path = output_dir / image_path.relative_to(input_dir)
            output_image_path = output_image_path.with_suffix('.pt')  # replace .png with .pt
            if not output_image_path.exists():
                self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        return image, str(image_path)
    

def process_and_save_images(input_dir, output_dir, batch_size):
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".png"):  # or whatever your image file extension is
                image_paths.append(Path(root) / file)

    dataset = ImageDataset(image_paths, transform, output_dir, input_dir)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    for input_images, batch_image_paths in tqdm(dataloader):
        input_images = input_images.to('cuda')  # Move to GPU

        # Run the model and save the output feature vectors
        feature_vectors = model(input_images)
        for image_path, feature_vector in zip(batch_image_paths, feature_vectors):
            output_image_path = output_dir / Path(image_path).relative_to(input_dir)
            output_image_path = output_image_path.with_suffix('.pt')  # replace .png with .pt

            # Skip if vector already exists
            if output_image_path.exists():
                continue

            # Ensure the output subdirectory exists
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the feature vector
            torch.save(feature_vector.cpu(), str(output_image_path))  # Move to CPU before saving

def augmented_process_and_save_images(input_dir, output_dir, batch_size):
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".png"):  # or whatever your image file extension is
                image_paths.append(Path(root) / file)

    dataset = ImageDataset(image_paths, transform, output_dir, input_dir)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    for augmented_images_batch, batch_image_paths in tqdm(dataloader):
        # Process each batch of augmented images
        for i, input_images in enumerate(augmented_images_batch):
            input_images = input_images.to('cuda')  # Move to GPU

            # Run the model and save the output feature vectors
            feature_vectors = model(input_images)
            for j, feature_vector in enumerate(feature_vectors):
                image_path = batch_image_paths[i]
                output_image_path = output_dir / Path(image_path).relative_to(input_dir)
                
                # Append the index of the augmentation to the filename
                output_image_path = output_image_path.with_stem(f"{output_image_path.stem}_{j}")
                output_image_path = output_image_path.with_suffix('.pt')

                # Skip if vector already exists
                if output_image_path.exists():
                    continue

                # Ensure the output subdirectory exists
                output_image_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save the feature vector
                torch.save(feature_vector.cpu(), str(output_image_path))  # Move to CPU before saving

            

augmented_process_and_save_images(Path("dataset/train"), Path("dataset/trainvecunpooled"), batch_size=8)

# image_path = "dataset/train/airplane/0_real/06215.png"
# image = Image.open(image_path)
# images = transform(image)
# print(len(images))

# to_pil = ToPILImage()

# def denormalize(tensor):
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
#     return tensor * std + mean


# # Iterate over each image tensor in the list
# for i, img_tensor in enumerate(images):
#     # Convert tensor to PIL image
#     img = to_pil(denormalize(img_tensor))
#     # img = img_tensor
#     # Define the output path for each image
#     output_path = f"{i}.png"
    
#     # Save the image
#     img.save(output_path)