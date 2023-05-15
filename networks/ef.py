import torch.nn as nn

class FeatureExtractionEfficientNet(nn.Module):
    def __init__(self, pretrained_model):
        super(FeatureExtractionEfficientNet, self).__init__()
        self.features = pretrained_model.features

    def forward(self, x):
        x = self.features(x)
        return x