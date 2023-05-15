import os
import csv
import torch

from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
from eval_config import *
import torchvision.models as models
import torch.nn as nn
from data import create_dataloader
from networks.trainer_new import AvgPoolClassifier
from util import prune_parallel_trained_model
from networks.ef import FeatureExtractionEfficientNet
from networks.trainer_vit import ViTNetwork, ViTNetworkImage

# Running tests
opt = TestOptions().parse(print_options=False)
model_name = os.path.basename(model_path).replace('.pth', '')
rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision']]

print("{} model testing on...".format(model_name))
for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)
    opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    opt.no_resize = True    # testing without resizing by default

    # model = resnet50(num_classes=1)
    if opt.vit:
        model = ViTNetworkImage()
        pre_model = None
        # pretrained_model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        # pre_model = FeatureExtractionEfficientNet(pretrained_model)
        # pre_model.cuda()
        # pre_model.eval()
    elif not opt.avg_pool_classifier:
        model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        model.classifier[0] = nn.Dropout(0.3, inplace=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        pre_model = None
    else:
        model = AvgPoolClassifier(1280, 1)
        pretrained_model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        pre_model = FeatureExtractionEfficientNet(pretrained_model)
        pre_model.cuda()
        pre_model.eval()
    
    state_dict = torch.load(model_path, map_location='cpu')
    new_state_dict = prune_parallel_trained_model(state_dict)
    
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    data_loader = create_dataloader(opt)
    acc, ap, _, _, _, _ = validate(model, opt, data_loader, pre_model=pre_model)
    rows.append([val, acc, ap])
    print("({}) acc: {}; ap: {}".format(val, acc, ap))

csv_name = results_dir + '/{}.csv'.format(model_name)
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
