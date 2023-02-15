from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import create_semisupervised_setting

import torch
import torchvision.transforms as transforms
import random


class MNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int = 0, known_outlier_class: int = 1, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0):
        super().__init__(root)
        # dataset = MNIST_Dataset(root=data_path,
        #                         normal_class=normal_class,
        #                         known_outlier_class=known_outlier_class,
        #                         n_known_outlier_classes=n_known_outlier_classes,
        #                         ratio_known_normal=ratio_known_normal,
        #                         ratio_known_outlier=ratio_known_outlier,
        #                         ratio_pollution=ratio_pollution)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
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
        train_set = MyMNIST(root=self.root, train=True, transform=transform, target_transform=target_transform,
                            download=True)

        # Create semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(train_set.targets.cpu().data.numpy(), self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)
        train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

        # Subset train_set to semi-supervised setup
        self.train_set = Subset(train_set, idx)

        # Get test set
        self.test_set = MyMNIST(root=self.root, train=False, transform=transform, target_transform=target_transform,
                                download=True)


# class MyMNIST(MNIST):
class MyMNIST(MNIST):
    """
    Torchvision MNIST class with additional targets for the semi-supervised setting and patch of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    def __init__(self, *args, **kwargs):
        #https://zhuanlan.zhihu.com/p/149532177
        super(MyMNIST, self).__init__(*args, **kwargs)
        #train_set = MyMNIST(root=self.root, train=True, transform=transform, target_transform=target_transform,
                            #download=True)

        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
        img, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])
        # print('img, target==========================', img.shape, target)
        # print('img, target==============================semi_target semi_target semi_target semi_target semi_target', semi_target)
        #img, target========================== torch.Size([28, 28]) 0
        #img, target==============================semi_target semi_target semi_target semi_target semi_target 0

        #img, target========================== torch.Size([28, 28]) 0 0

        #img, target========================== torch.Size([28, 28]) 0
# img, target========================== torch.Size([28, 28]) 1
# img, target========================== torch.Size([28, 28]) 2
# img, target========================== torch.Size([28, 28]) 3
# img, target========================== torch.Size([28, 28]) 4
# img, target========================== torch.Size([28, 28]) 5
# img, target========================== torch.Size([28, 28]) 6

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        # print('img, target, semi_target, index================', img.shape, target, semi_target, index)
        #img, target, semi_target, index================ torch.Size([1, 28, 28]) 1 0 9999


        return img, target, semi_target, index
