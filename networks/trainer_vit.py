import torch
import torch.nn as nn
from networks.base_model import BaseModel, init_weights
from accelerate import Accelerator
import torchvision.models as models


import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from vit_pytorch import SimpleViT
from einops import rearrange
from util import prune_parallel_trained_model
from networks.ef import FeatureExtractionEfficientNet

accelerator = Accelerator()

class ViTNetwork(nn.Module):
    def __init__(self, pretrained_model_name='vit_base_patch16_224', num_classes=1):
        super(ViTNetwork, self).__init__()

        # load pretrained ViT model
        self.vit_model = timm.create_model(pretrained_model_name, pretrained=True)

        # remove the patch embedding layer
        self.feature_transform = nn.Linear(1280, 768)
        self.vit_model.patch_embed = nn.Identity()

        # adjust the size of positional embeddings
        embed_dim = self.vit_model.pos_embed.shape[-1]  # keep the same embed_dim as in the pretrained model
        self.vit_model.pos_embed = nn.Parameter(torch.zeros(1, 226, 768))

        # replace the classification head
        self.vit_model.head = nn.Linear(self.vit_model.head.in_features, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1)  # rearrange the input to be suitable for ViT
        x = self.feature_transform(x)
        x = self.vit_model(x)  # pass the tokens through the ViT model
        return x
    
class ViTNetworkImage(nn.Module):
    def __init__(self, pretrained_model_name='vit_base_patch16_224', num_classes=1):
        super(ViTNetwork, self).__init__()

        # load pretrained ViT model
        self.vit_model = timm.create_model(pretrained_model_name, pretrained=True)
        state_dict = torch.load("./weights/effv2.pth")
        new_state_dict = prune_parallel_trained_model(state_dict)
        
        ef_model = models.efficientnet_v2_m()
        ef_model.classifier[0] = nn.Dropout(0.3, inplace=True)
        ef_model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        ef_model.load_state_dict(new_state_dict)
        self.ef_model = FeatureExtractionEfficientNet(ef_model)

        # remove the patch embedding layer
        self.feature_transform = nn.Linear(1280, 768)
        self.vit_model.patch_embed = nn.Identity()

        # adjust the size of positional embeddings
        embed_dim = self.vit_model.pos_embed.shape[-1]  # keep the same embed_dim as in the pretrained model
        self.vit_model.pos_embed = nn.Parameter(torch.zeros(1, 226, 768))

        # replace the classification head
        self.vit_model.head = nn.Linear(self.vit_model.head.in_features, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.ef_model(x)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1)  # rearrange the input to be suitable for ViT
        x = self.feature_transform(x)
        x = self.vit_model(x)  # pass the tokens through the ViT model
        return x


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = ViTNetworkImage()

        if not self.isTrain or opt.continue_train:
            self.model = ViTNetworkImage()
            self.load_networks(opt.epoch)
            
        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()

            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        self.model, self.optimizer = accelerator.prepare(self.model, self.optimizer)

    
    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        self.input = accelerator.gather(input[0])
        self.label = accelerator.gather(input[1]).float()


    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        accelerator.backward(self.loss)
        self.optimizer.step()