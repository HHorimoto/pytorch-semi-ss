import torch
from torchvision import datasets

from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomCrop, Pad, Normalize
from PIL import Image
import pathlib
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

from src.utils.seeds import worker_init_fn

class UnsupervisedDataset(Dataset):
    def __init__(self, dataset, transform_1, transform_2):
        self.dataset = dataset
        self.transform_1 = transform_1
        self.transform_2 = transform_2

    def __getitem__(self, index):
        image, label = self.dataset[index]

        image_1 = self.transform_1(image)
        image_2 = self.transform_2(image)

        return image_1, image_2

    def __len__(self):
        return len(self.dataset)

def x_u_split(dataset, test_size=49000, seed=42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed) # same percentage of samples
    sss = sss.split(list(range(len(dataset))), dataset.targets)
    label_idxs, unlabel_idxs = next(sss)

    return label_idxs, unlabel_idxs

def create_dataset(root='./data/', download=True, batch_size=16):

    transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    unsup_transform = Compose([
        RandomHorizontalFlip(), Pad(4, padding_mode="reflect"), RandomCrop(32), 
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
    unsup_trainset = datasets.CIFAR10(root=root, train=True,  download=download, transform=None)
    testset = datasets.CIFAR10(root=root, train=False, download=download, transform=transform)

    label_idxs, unlabel_idxs = x_u_split(trainset)

    supset = Subset(trainset, label_idxs)
    train_labels = [trainset.targets[idx] for idx in label_idxs]
    supset.train_labels = train_labels

    unsupset = Subset(unsup_trainset, unlabel_idxs)
    unsupset = UnsupervisedDataset(unsupset, unsup_transform, unsup_transform)

    unsup_batch_size = int(batch_size * (len(unsupset)/len(supset)))

    sup_loader = DataLoader(supset, batch_size, shuffle=True, num_workers=2, 
                              pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
    unsup_loader = DataLoader(unsupset, unsup_batch_size, shuffle=True, num_workers=2, 
                              pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
    test_loader  = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, 
                              pin_memory=True, worker_init_fn=worker_init_fn, drop_last=False)
    
    return sup_loader, unsup_loader, test_loader