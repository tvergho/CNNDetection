import os
import csv
import torch

from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
from eval_config import *
import torchvision.models as models
import torch.nn as nn
from data import create_dataloader, ImageDataset, LimitedImageDataset
from networks.trainer_new import AvgPoolClassifier
from util import prune_parallel_trained_model
from torch import Tensor, autocast, inference_mode
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from PIL import Image

# Running tests
opt = TestOptions().parse(print_options=False)
model_name = os.path.basename(model_path).replace('.pth', '')
rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision']]

transform = transforms.Compose([
    transforms.Lambda(lambda img: TF.resize(img, opt.loadSize, interpolation=Image.BILINEAR)),
    transforms.RandomCrop(opt.cropSize),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("{} model testing on...".format(model_name))
for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)
    # opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    opt.no_resize = True    # testing without resizing by default

    # model = resnet50(num_classes=1)
    if opt.vit:
        model = ViTNetworkImage()
        pre_model = None
    elif opt.avg_pool_classifier:
        model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        model.classifier[0] = nn.Dropout(0.3, inplace=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        pre_model = None
    else:
        # model = models.efficientnet_v2_s()
        model = models.efficientnet_v2_m()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        pre_model = None
    
    state_dict = torch.load(model_path, map_location='cpu')
    new_state_dict = prune_parallel_trained_model(state_dict)
    
    # model.load_state_dict(state_dict['model'])
    model.load_state_dict(new_state_dict)
    # model.cuda()
    mps_device = torch.device("mps")
    model = model.to(mps_device)
    model.eval()

    local_dataset = ImageDataset(opt.dataroot, transform=transform)
    data_loader = DataLoader(local_dataset, batch_size=opt.batch_size, shuffle=True)
    print(opt.dataroot, len(local_dataset))
    # data_loader = create_dataloader(opt)
    acc, ap, r_acc, f_acc, _, _ = validate(model, opt, data_loader, pre_model=pre_model)
    rows.append([val, acc, ap])
    print("({}) acc: {}; ap: {}; r_acc: {}; f_acc: {}".format(val, acc, ap, r_acc, f_acc))

csv_name = results_dir + '/{}.csv'.format(model_name)
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
