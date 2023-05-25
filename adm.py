import torch as th

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)

import torchvision.transforms as transforms
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP

def normalize(tensor, min_val, max_val, gpu_id=0):
    tensor_min = tensor.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0].to(f'cuda:{gpu_id}')
    tensor_max = tensor.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0].to(f'cuda:{gpu_id}')

    tensor_norm = (tensor - tensor_min) * (max_val - min_val) / (tensor_max - tensor_min) + min_val
    return tensor_norm.to(f'cuda:{gpu_id}')


class DiffusionModel:
    def __init__(self,
        gpu_id=0,
        model_path="/workspace/guided-diffusion/256x256_diffusion_uncond.pt", 
        clip_denoised=True, 
        batch_size=8, 
        image_size=256):
        
        self.clip_denoised = clip_denoised
        self.batch_size = batch_size
        self.image_size = image_size
        
        self.model, self.diffusion = create_model_and_diffusion(
            **model_and_diffusion_defaults()
        )
        self.model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"))
        self.model.eval()
        # self.device = th.device(f"cuda:{gpu_id}" if th.cuda.is_available() else "cpu")
        self.model.to("cuda")
        self.gpu_id = gpu_id
        
        self.model = DDP(self.model)

    def get_image_dire(self, image):
        # img = normalize(image, -1, 1)
        sample_fn = self.diffusion.ddim_reverse_sample_loop
        sample = sample_fn(
            self.model.module,
            (self.batch_size, 3, self.image_size, self.image_size),
            clip_denoised=self.clip_denoised,
            model_kwargs={},
            progress=True,
            noise=image,
        )

        noise_ = sample
        sample_fn = self.diffusion.ddim_sample_loop
        sample = sample_fn(
            self.model.module,
            (self.batch_size, 3, self.image_size, self.image_size),
            clip_denoised=self.clip_denoised,
            model_kwargs={},
            progress=True,
            noise=noise_,
        )

        decoded_image1 = sample.clone()
        decoded_image2 = image.clone()

        decoded_image1 = (decoded_image1).clamp(0, 1)
        decoded_image2 = (decoded_image2).clamp(0, 1)
        # decoded_image1 = normalize(decoded_image1, -1, 1, gpu_id=self.gpu_id)
        # decoded_image2 = normalize(decoded_image2, -1, 1, gpu_id=self.gpu_id)

        decoded_image1 = decoded_image1.contiguous()
        decoded_image2 = decoded_image2.contiguous()

        dire = th.abs(decoded_image1 - decoded_image2)
        dire = (dire * 255).clamp(0, 255).to(th.uint8)
        # dire = ((dire + 1) * 127.5).clamp(0, 255).to(th.uint8)
        return dire, decoded_image1, decoded_image2

    
transform = transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

if __name__ == "__main__":
    img_path = "/workspace/CNNDetection/dataset/fake/imagenet1k/006_stingray/006_1704.jpg"
    image = Image.open(img_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).cuda()
    
    model = DiffusionModel()
    print(model.get_image_dire(image))