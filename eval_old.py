import os
import csv
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from networks.resnet import resnet50
from options.test_options import TestOptions
from eval_config import *
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from tqdm import tqdm
from data import numpy_to_pil_image, CombinedLimitedDataset
import torchvision.models as models
import torch.nn as nn
from networks.trainer_vit import ViTNetworkImage
from util import prune_parallel_trained_model

threshold = 0.5

# Running tests
opt = TestOptions().parse(print_options=False)
model_name = os.path.basename(model_path).replace('.pth', '')
rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision']]

print("{} model testing on...".format(model_name))
# model = resnet50(num_classes=1)
model = ViTNetworkImage()
# model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

state_dict = torch.load(model_path, map_location='cpu')
new_state_dict = prune_parallel_trained_model(state_dict)
 
model.load_state_dict(new_state_dict)
# model.cuda()
mps_device = torch.device("mps")
model = model.to(mps_device)
model.eval()

real_dataset_name = 'imagenet-1k'
real_dataset = load_dataset(real_dataset_name, split='train', use_auth_token=True, streaming=True)

fake_dataset_name = 'poloclub/diffusiondb'
fake_dataset = load_dataset(fake_dataset_name, split='train', use_auth_token=True, streaming=True)

transform = transforms.Compose([
    transforms.Lambda(lambda img: TF.resize(img, opt.loadSize, interpolation=Image.BILINEAR)),
    transforms.RandomCrop(opt.cropSize),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = CombinedLimitedDataset(fake_dataset, real_dataset, transform=transform, max_size=1000)
data_loader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=0, shuffle=True, pin_memory=True)

rows = []
with torch.no_grad():
    y_true, y_pred = [], []
    for img, label in tqdm(data_loader):
        # in_tens = img.cuda()
        in_tens = img.to(mps_device)
        y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
        y_true.extend(label.flatten().tolist())
        
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # y_true = 1 - y_true

    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > threshold)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > threshold)
    acc = accuracy_score(y_true, y_pred > threshold)
    ap = average_precision_score(y_true, y_pred)
    print("({}) acc: {}; ap: {}; r_acc: {}; f_acc: {}".format(model_name, acc, ap, r_acc, f_acc))
    rows.append([acc, ap, r_acc, f_acc])

csv_name = results_dir + '/{}.csv'.format(model_name)
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
