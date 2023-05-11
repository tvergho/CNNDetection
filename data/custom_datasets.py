from PIL import Image
from torch.utils.data import Dataset
import numpy as np

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