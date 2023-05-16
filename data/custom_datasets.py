from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path

def numpy_to_pil_image(img):
    img = np.asarray(img)        
    if len(img.shape) == 2:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    
    try:
      return Image.fromarray(img)
    except:
      img = (img * 255).astype(np.uint8)
      return Image.fromarray(img)

class CombinedLimitedDataset(Dataset):
    def __init__(self, dataset_1, dataset_2, transform=None, max_size=None, dataset_1_is_local=False):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.transform = transform
        self.iter1 = iter(self.dataset_1)
        self.iter2 = iter(self.dataset_2)
        self.max_size = max_size
        self.dataset_1_is_local = dataset_1_is_local

    def __len__(self):
        return self.max_size

    def __getitem__(self, idx):
        if idx < int(self.max_size / 2):
          if not self.dataset_1_is_local:
            try:
              img = next(self.iter1)['image']
              label = 1
            except StopIteration:
              self.iter1 = iter(self.dataset_1)
              img = next(self.iter1)['image']
              label = 1
          else:
            img, label = self.dataset_1[idx]
        else:
            try:
              img = next(self.iter2)['image']
              label = 0
            except StopIteration:
              self.iter2 = iter(self.dataset_2)
              img = next(self.iter2)['image']
              label = 0

        img = numpy_to_pil_image(img)

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

class LocalDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        image_paths = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        return img, 1  # "fake" class label

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.image_files = list(self.root_dir.glob('**/*.jpg')) + list(self.root_dir.glob('**/*.jpeg')) + list(self.root_dir.glob('**/*.png'))
        self.labels = [str(file.parent.name) for file in self.image_files]

        self.label_to_int = {
            '0_real': 0,
            '1_fake': 1
        }
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        label = self.label_to_int[img_path.parent.name]
        return img, label

    def __len__(self):
        return len(self.image_files)
