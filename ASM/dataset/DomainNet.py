
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image
from torch.utils import data


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
        T.append(torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        transforms = torchvision.transforms.Compose(T)
    else:
        T.append(torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        self.transform = torchvision.transforms.Compose(
            initialize_transform(set, self.img_size))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        sample = Image.open(osp.join(self.root, img_path)
                            ).convert('RGB')  # H,W,C
        sample.resize((1280, 720), Image.BICUBIC)
        target = self.labels[index]
        target = np.asarray(target, np.float32)
        if self.transform is not None:
            sample = self.transform(sample)
        sample = np.asarray(sample, np.float32)
        size = sample.shape
        #print("$$$$$$$$$$$$$$$$$$", size)
        return sample, target, np.array(size), img_path


if __name__ == '__main__':
    dst = DomainNet("/home/skadam/workspace/datasets/domainnet/",
                    "/home/skadam/workspace/datasets/domainnet/clipart_train.txt")
    trainloader = data.DataLoader(dst, batch_size=16,shuffle=True)
    for i, data in enumerate(trainloader):
        imgs, labels, _, _ = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            t = (str(np.array(labels)))
            plt.text(4, 1, t, ha='left', rotation=0, wrap=True)
            plt.show()
        

