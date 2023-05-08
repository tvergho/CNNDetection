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

def numpy_to_pil_image(img):
    img = np.asarray(img)        
    if len(img.shape) == 2:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

class CombinedLimitedDataset(Dataset):
    def __init__(self, dataset_1, dataset_2, transform=None, max_size=None):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.transform = transform
        self.iter1 = iter(self.dataset_1)
        self.iter2 = iter(self.dataset_2)
        self.max_size = max_size

    def __len__(self):
        return self.max_size

    def __getitem__(self, idx):
        if idx < int(self.max_size / 2):
            img = next(self.iter1)
            label = 0
        else:
            img = next(self.iter2)
            label = 1

            
        img = numpy_to_pil_image(img['image'])

        if self.transform:
            img = self.transform(img)

        return img, label

    
class LimitedDataset(Dataset):
    def __init__(self, dataset, max_size):
        self.dataset = dataset
        self.max_size = min(max_size, len(dataset))

    def __len__(self):
        return self.max_size

    def __getitem__(self, idx):
        return self.dataset[idx]

# Running tests
opt = TestOptions().parse(print_options=False)
model_name = os.path.basename(model_path).replace('.pth', '')
rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision']]

print("{} model testing on...".format(model_name))
model = resnet50(num_classes=1)
state_dict = torch.load(model_path, map_location='cpu')
if hasattr(state_dict, '_metadata'):
    del state_dict._metadata

# Remove the 'module' prefix
new_state_dict = {}
for key, value in state_dict['model'].items():
    new_key = key.replace('module.', '')  # Remove 'module.' from the key
    new_state_dict[new_key] = value 
model.load_state_dict(new_state_dict)
model.cuda()
model.eval()

real_dataset_name = 'imagenet-1k'
real_dataset = load_dataset(real_dataset_name, split='train', use_auth_token=True, streaming=True)

fake_dataset_name = 'poloclub/diffusiondb'
fake_dataset = load_dataset(fake_dataset_name, split='train', use_auth_token=True, streaming=True)

crop_func = transforms.RandomCrop(opt.cropSize)
transform = transforms.Compose([
                transforms.Lambda(lambda img: TF.resize(img, opt.loadSize, interpolation=Image.BILINEAR)),
                crop_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
dataset = CombinedLimitedDataset(fake_dataset, real_dataset, transform=transform, max_size=1000)
data_loader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=0, shuffle=True, pin_memory=True)

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
    print("({}) acc: {}; ap: {}".format(model_name, acc, ap))
    
#     acc, ap, _, _, _, _ = validate(model, opt)
#     rows.append([val, acc, ap])
#     

# csv_name = results_dir + '/{}.csv'.format(model_name)
# with open(csv_name, 'w') as f:
#     csv_writer = csv.writer(f, delimiter=',')
#     csv_writer.writerows(rows)
