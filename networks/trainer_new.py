import torch
import torch.nn as nn
from networks.base_model import BaseModel, init_weights
from accelerate import Accelerator
import torchvision.models as models

accelerator = Accelerator()

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)
    
class AvgPoolClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.3):
        super(AvgPoolClassifier, self).__init__()
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob, inplace=True),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1) # Flatten the tensor before passing it to the classifier
        return self.classifier(x)

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = AvgPoolClassifier(1280, 1)
            torch.nn.init.normal_(self.model.classifier[1].weight.data, 0.0, opt.init_gain)

        if not self.isTrain or opt.continue_train:
            self.model = AvgPoolClassifier(1280, 1)
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