import functools
import torch
import torch.nn as nn
from networks.resnet import resnet50
from networks.base_model import BaseModel, init_weights
from accelerate import Accelerator
from efficientnet_pytorch import EfficientNet

accelerator = Accelerator()

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            # self.model = resnet50(pretrained=True)
            self.model = EfficientNet.from_pretrained('efficientnet-b5')
             # freeze all layers
            for param in self.model.parameters():
                param.requires_grad = False

            # replace the last layer
            self.model._fc = nn.Linear(2048, 1)

            # unfreeze the last layer
            for param in self.model._fc.parameters():
                param.requires_grad = True

            torch.nn.init.normal_(self.model._fc.weight.data, 0.0, opt.init_gain)

        if not self.isTrain or opt.continue_train:
            self.model = resnet50(num_classes=1)
            
        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)

        self.model, self.optimizer = accelerator.prepare(
            self.model, self.optimizer
        )


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

