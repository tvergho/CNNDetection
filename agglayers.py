def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )

    def forward(self, x):
        primitives = []
        i = 0
        for layer in self.backbone:
            x = layer(x)
            if isinstance(layer, MBConvBlock):  # or Fused-MBConv
                kernel_size = 14 if i < 19 else 7
                pooled = F.avg_pool2d(x, kernel_size)  # kernel_size=14 for low level, 7 for high level
                flattened = pooled.view(pooled.size(0), -1)
                primitive = self.mlp(flattened)
                primitives.append(primitive)
                i += 1
        return torch.stack(primitives, dim=-1)

transform = transforms.Compose([
                transforms.Lambda(lambda img: TF.resize(img, crop_size, interpolation=Image.BILINEAR)),
                transforms.Lambda(lambda img: data_augment(img, opt)),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])