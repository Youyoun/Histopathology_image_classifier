from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, manifest, transforms=None):
        super().__init__()
        self.manifest = manifest
        with open(self.manifest) as f:
            data = [x.split(",") for x in f.read().splitlines()]
        self.images_paths = [d[0] for d in data]
        self.labels = [int(d[1]) for d in data]
        self.transform = transforms

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(self.images_paths[idx])

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = TF.to_tensor(img)

        return img, self.labels[idx]


class TestDataset(Dataset):
    def __init__(self, manifest, transforms=None):
        super().__init__()
        self.manifest = manifest
        with open(self.manifest) as f:
            data = [x.split(",") for x in f.read().splitlines()]
        self.images_paths = [d[0] for d in data]
        self.transform = transforms

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(self.images_paths[idx])

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = TF.to_tensor(img)

        return img


data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
data_transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
