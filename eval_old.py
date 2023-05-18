import os
import csv
import torch

from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
from eval_config import *
import torchvision.models as models
import torch.nn as nn
from data import create_dataloader, ImageDataset
from networks.trainer_new import AvgPoolClassifier
from util import prune_parallel_trained_model
from diffusers import LDMPipeline, UNet2DModel, VQModel, DDIMScheduler
from torch import Tensor, autocast, inference_mode
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

unet = UNet2DModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="unet", torch_dtype=torch.float16).to("cuda")
vqvae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae", torch_dtype=torch.float16).to("cuda")
scheduler = DDIMScheduler.from_pretrained("CompVis/ldm-celebahq-256", subfolder="scheduler")

@torch.compile
def get_image_dire(image):
    with autocast("cuda"), torch.no_grad():
        latents = vqvae.encode(image.half()).latents

        scheduler.set_timesteps(20)
        for i, e in enumerate(np.flip(scheduler.timesteps, 0)):
            latents = scheduler.reverse_step(unet(latents, e).sample, e, latents).next_sample

        latents_ = latents.clone()

        for i, e in enumerate(scheduler.timesteps):
            latents_ = scheduler.step(unet(latents_, e).sample, e, latents_).prev_sample

        dire = calculate_dire(image.half(), latents_)
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

# Running tests
opt = TestOptions().parse(print_options=False)
model_name = os.path.basename(model_path).replace('.pth', '')
rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision']]

print("{} model testing on...".format(model_name))
for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)
    # opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    opt.no_resize = True    # testing without resizing by default

    # model = resnet50(num_classes=1)
    if opt.vit:
        model = ViTNetworkImage()
        pre_model = None
        # pretrained_model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        # pre_model = FeatureExtractionEfficientNet(pretrained_model)
        # pre_model.cuda()
        # pre_model.eval()
    elif opt.avg_pool_classifier:
        model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        model.classifier[0] = nn.Dropout(0.3, inplace=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        pre_model = None
    else:
        # model = AvgPoolClassifier(1280, 1)
        model = models.efficientnet_v2_s()
        # model.classifier[0] = nn.Dropout(0.3, inplace=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        # pre_model = get_image_dire
        pre_model = None
    
    state_dict = torch.load(model_path, map_location='cpu')
    new_state_dict = prune_parallel_trained_model(state_dict)
    
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    local_dataset = ImageDataset(dataroot, transform=transforms.ToTensor())
    data_loader = DataLoader(local_dataset, batch_size=opt.batch_size)
    # data_loader = create_dataloader(opt)
    acc, ap, _, _, _, _ = validate(model, opt, data_loader, pre_model=pre_model)
    rows.append([val, acc, ap])
    print("({}) acc: {}; ap: {}".format(val, acc, ap))

csv_name = results_dir + '/{}.csv'.format(model_name)
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
