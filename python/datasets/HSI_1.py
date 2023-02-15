from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import create_semisupervised_setting

import torch
import torchvision.transforms as transforms
import random

from scipy import io, misc
import os
import re
import spectral
import numpy as np


class HSI_Dataset_1(TorchvisionDataset):
    def __init__(self, cfg, root: str, normal_class: int = 0,  dataset_name: str = 'HYDICE_urban',
                 known_outlier_class: int = 1, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0,
                 random_state: int = 1, patch_size: int = 1):
        super().__init__(root)

        # Define normal and outlier classes
        self.cfg = cfg
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        # self.outlier_classes = list(range(0, 10))#可以改为1,2，。。。，10等
        self.outlier_classes = list(range(0, 2))#可以改为1,2，。。。，10等
        #[0, 1]
        self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        elif n_known_outlier_classes == 1:
            self.known_outlier_classes = tuple([known_outlier_class])
        else:
            self.known_outlier_classes = tuple(random.sample(self.outlier_classes, n_known_outlier_classes))
            #random.sample，多用于截取列表的指定长度的随机数，但不改变列表本身的顺序


        # MNIST preprocessing: feature scaling to [0, 1]
        transform = transforms.ToTensor()
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # Get train set
        train_set = MyHSI_1(cfg=cfg, root=root, dataset_name = 'HYDICE_urban', train=True, transform=transform, target_transform=target_transform, download=True,patch_size=patch_size)
        #(self, root: str, normal_class: int = 0,  dataset_name: str = 'HYDICE_urban',
                 # known_outlier_class: int = 1, n_known_outlier_classes: int = 0,
                 # ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0,
                 # ratio_pollution: float = 0.0,
                 # random_state: int = 1):
        #data, label, semi_target, i
        #sample, target, semi_target================= torch.Size([175]) 0 0
        #sample
        # Create semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(train_set.targets.cpu().data.numpy(), self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)
        train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

        # Subset train_set to semi-supervised setup
        self.train_set = Subset(train_set, idx)

        # Get test set
        self.test_set = MyHSI_1(cfg=cfg, root=self.root, dataset_name = 'HYDICE_urban', train=False, transform=transform, target_transform=target_transform, download=True)
        
######################################################### 
def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    # print('dataset==================', dataset)
    # dataset================== ./Datasets/PaviaU/PaviaU.mat
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return misc.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))
#########################################################

# class MyMNIST(MNIST):
class MyHSI_1(torch.utils.data.Dataset):
    """
    Torchvision MNIST class with additional targets for the semi-supervised setting and patch of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    # def __init__(self, *args, **kwargs ):
    def __init__(self, cfg, root, dataset_name, train, transform, target_transform, download, patch_size):
        #https://zhuanlan.zhihu.com/p/149532177
        super(MyHSI_1, self).__init__()
        #self.test_set = MyHSI_1(cfg=cfg, root=self.root, dataset_name = 'HYDICE_urban', train=False, transform=transform, target_transform=target_transform, download=True)

        self.semi_targets = torch.zeros_like(self.targets)

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.patch_size = patch_size

        self.ignored_labels = set(cfg.settings['ignored_labels'])
        self.flip_augmentation = cfg.settings['flip_augmentation']
        self.radiation_augmentation = cfg.settings['radiation_augmentation']
        self.mixture_augmentation = cfg.settings['mixture_augmentation']
        self.center_pixel = cfg.settings['center_pixel']
        supervision = cfg.settings['supervision']

        #mat = loadmat(self.root)
        img = open_file(self.root + dataset_name + '.mat')['hsi']
        gt = open_file(self.root + dataset_name + '.mat')['hsi_gt']
        # img = mat['hsi']
        # gt = mat['hsi_gt']

        img = np.asarray(img, dtype='float32')
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        mask = np.ones_like(gt)
        # numpy的ones_like函数返回一个用1填充的跟输入数组形状和类型一样的数组
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        # patch_size默认为9
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if x > p
                                 and x < img.shape[0] - p and y > p and y < img.shape[1] - p])
        self.labels = [gt[x, y] for x, y in self.indices]
        self.data = img
        self.label = gt



    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert (self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise




    def __getitem__(self, i):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.1:
            data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.2:
            data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)
        # print('data, label=====================', data.shape, label.shape)
        # data, label===================== torch.Size([1, 103, 7, 7]) torch.Size([])
        #print('data, label=====================', data.shape, label, len(self.indices))
        # data, label===================== torch.Size([1, 103, 7, 7]) tensor(2)
        semi_target = int(self.semi_targets[i])
        semi_target = torch.from_numpy(semi_target)
        sample = data
        target = label
        index = i
        # return img, target, semi_target, i
        #sample, target, semi_target, index
        #return data, label, semi_target, i
        return sample, target, semi_target, index

    def __len__(self):
        return len(self.indices)

