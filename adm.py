import torch as th

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)

import torchvision.transforms as transforms
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP

class DiffusionModel:
    def __init__(self, 
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
        self.model.cuda()
        
        self.model = DDP(self.model)

    def get_image_dire(self, image):
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

        decoded_image1 = sample
        decoded_image2 = image

        decoded_image1 = (decoded_image1 / 2 + 0.5).clamp(0, 1)
        decoded_image2 = (decoded_image2 / 2 + 0.5).clamp(0, 1)

        decoded_image1 = decoded_image1.contiguous()
        decoded_image2 = decoded_image2.contiguous()

        dire = th.abs(decoded_image1 - decoded_image2)
        dire = (dire * 255).clamp(0, 255).to(th.uint8)
        return dire

    
transform = transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.ToTensor(),
        ])

if __name__ == "__main__":
    img_path = "/workspace/CNNDetection/dataset/fake/imagenet1k/006_stingray/006_1704.jpg"
    image = Image.open(img_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).cuda()
    
    model = DiffusionModel()
    print(model.get_image_dire(image))