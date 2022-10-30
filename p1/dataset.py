import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
train_tfm = transforms.Compose(
    [
        transforms.RandomResizedCrop((64, 64), (0.8, 1.25), (0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


class UnNormalize(object):
    def __init__(self):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class FaceDataset(Dataset):
    def __init__(
        self,
        path,
        tfm=train_tfm,
    ):
        super(FaceDataset).__init__()
        self.path = path
        self.files = sorted(
            [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".png")]
        )
        print(f"One {path} sample", self.files[0])
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        return im
