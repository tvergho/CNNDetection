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

from util import prune_parallel_trained_model

# Running tests
opt = TestOptions().parse(print_options=False)
model_name = os.path.basename(model_path).replace('.pth', '')
rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision']]

print("{} model testing on...".format(model_name))
model = resnet50(num_classes=1)
state_dict = torch.load(model_path, map_location='cpu')
new_state_dict = prune_parallel_trained_model(state_dict)
 
model.load_state_dict(new_state_dict)
model.cuda()
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
        in_tens = img.cuda()
        y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
        y_true.extend(label.flatten().tolist())
        
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # y_true = 1 - y_true

    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    print("({}) acc: {}; ap: {}; r_acc: {}; f_acc: {}".format(model_name, acc, ap, r_acc, f_acc))
    rows.append([val, acc, ap, r_acc, f_acc])

csv_name = results_dir + '/{}.csv'.format(model_name)
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
