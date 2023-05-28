from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path
import random

def numpy_to_pil_image(img):
    img = np.array(img)        
    if len(img.shape) == 2:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    
    try:
      return Image.fromarray(img)
    except:
      img = (img * 255).astype(np.uint8)
      return Image.fromarray(img)

class CombinedLimitedDataset(Dataset):
    def __init__(self, dataset_1, dataset_2, transform=None, max_size=None, dataset_1_is_local=False, dataset_2_is_local=False):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.transform = transform
        self.iter1 = iter(self.dataset_1)
        self.iter2 = iter(self.dataset_2)
        self.max_size = max_size
        self.dataset_1_is_local = dataset_1_is_local
        self.dataset_2_is_local = dataset_2_is_local

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
            if not self.dataset_2_is_local:
                try:
                    img = next(self.iter2)['image']
                    label = 0
                except StopIteration:
                    self.iter2 = iter(self.dataset_2)
                    img = next(self.iter2)['image']
                    label = 0
            else:
                img, label = self.dataset_2[idx - int(self.max_size / 2)]

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

class FolderDataset(Dataset):
    def __init__(self, root_dir, label=1):
        self.root_dir = root_dir
        self.image_paths = self._get_image_paths()
        self.label = label

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

        return img, self.label

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_size=None):
        self.root_dir = Path(root_dir)
        self.image_files = list(self.root_dir.glob('**/*.jpg')) + list(self.root_dir.glob('**/*.jpeg')) + list(self.root_dir.glob('**/*.png'))
        self.labels = [str(file.parent.name) for file in self.image_files]

        self.label_to_int = {
            '0_real': 0,
            '1_fake': 1
        }
        self.transform = transform
        self.max_size = max_size

        # Shuffle image files and labels in unison
        if self.max_size:
            self.image_files, self.labels = self._shuffle_files_and_labels()

    def _shuffle_files_and_labels(self):
        zipped = list(zip(self.image_files, self.labels))
        random.shuffle(zipped)
        return zip(*zipped)

    def __getitem__(self, idx):
        try:
            img_path = self.image_files[idx]
            img = Image.open(img_path).convert('RGB')

            if self.transform:
                img = self.transform(img)

            label = self.label_to_int[img_path.parent.name]
        except Exception as e:
            print(f"Exception occurred for index {idx}: {e}")
            random_idx = random.choice(range(len(self.image_files)))
            return self.__getitem__(random_idx)

        return img, label

    def __len__(self):
        if self.max_size:
            return min(self.max_size, len(self.image_files))
        else:
            return len(self.image_files)

class LimitedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Initialize empty list for image files and labels
        self.image_files = []
        self.labels = []

        self.label_to_int = {
            '0_real': 0,
            '1_fake': 1
        }

        # Find minimum number of images across classes
        min_images = min(len(list(self.root_dir.glob('**/0_real/*.png'))), 
                         len(list(self.root_dir.glob('**/1_fake/*.png'))))

        # Load equal number of images from each class
        for label, int_label in self.label_to_int.items():
            class_files = list(self.root_dir.glob(f'**/{label}/*.png'))[:min_images]
            self.image_files += class_files
            self.labels += [int_label]*len(class_files)

    def __getitem__(self, idx):
        try:
            img_path = self.image_files[idx]
            img = Image.open(img_path).convert('RGB')

            if self.transform:
                img = self.transform(img)

            label = self.labels[idx]
        except Exception as e:
            print(f"Exception occurred for index {idx}: {e}")
            random_idx = random.choice(range(len(self.image_files)))
            return self.__getitem__(random_idx)

        return img, label

    def __len__(self):
        return len(self.image_files)
