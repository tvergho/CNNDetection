import torch
import torchvision.models as models

# model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
# print(model)
arr = torch.load("dataset/trainvecunpooled/airplane/0_real/07416_0.pt")
print(arr.size())