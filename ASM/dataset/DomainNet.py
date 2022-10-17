
import torch
import random
import torchvision
import numpy as np
from PIL import Image
import os.path as osp
from torch.utils import data
from torchvision import transforms


def loadTxt(txt_path):

    img_path = []
    labels = []

    with open(txt_path, 'r') as data_file:
        for line in data_file:
            data = line.split()
            img_path.append(data[0])
            labels.append(data[1])

    return img_path, np.asarray(labels, dtype=np.int64)

def initialize_transform(set, img_size):
    T = []
    if set == "train":
            T.append(torchvision.transforms.Resize(img_size))
            T.append(torchvision.transforms.RandomResizedCrop(img_size))
            T.append(torchvision.transforms.RandomHorizontalFlip())
            T.append(torchvision.transforms.ToTensor())
            T.append(torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            transforms = torchvision.transforms.Compose(T)
    else:
        T.append(torchvision.transforms.Compose([
                                            torchvision.transforms.Resize(img_size),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ]))
    return T



class DomainNet(data.Dataset):
    def __init__(self, root_path, list_path, img_size=(224, 224), crop_size=(321, 321), transform=None, set="train"):
        self.root = root_path
        self.list_path = list_path
        self.crop_w, self.crop_h = crop_size
        self.img_size = img_size
        self.transform = transform
        self.imgs, self.labels = loadTxt(list_path)
        self.transform = torchvision.transforms.Compose(initialize_transform(set, self.img_size))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        sample = Image.open(osp.join(self.root, img_path)
                            ).convert('RGB')  # H,W,C
        sample = sample.resize((1280,720), Image.BICUBIC)
        target = self.labels[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, np.array(self.img_size), img_path
