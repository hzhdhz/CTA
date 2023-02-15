from pathlib import Path
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision.datasets.utils import download_url

import os
import torch
import numpy as np


class ODDSDataset(Dataset):
    """
    ODDSDataset class for datasets from Outlier Detection DataSets (ODDS): http://odds.cs.stonybrook.edu/

    Dataset class with additional targets for the semi-supervised setting and modification of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    def __init__(self, root: str, dataset_name: str, train=True, random_state=None, download=False):
        super(Dataset, self).__init__()

        self.classes = [0, 1]

        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.train = train  # training set or test set
        self.file_name = self.dataset_name + '.mat'
        self.data_file = self.root / self.file_name

        mat = loadmat(self.data_file)
        X = mat['X']
        y = mat['y'].ravel()
        mask = mat['mask'].ravel()
        # idx_norm = y == 0
        # idx_out = y == 1

        idx_norm_mask = mask == 2
        idx_out_mask = mask == 1

        X_train_norm = X[idx_norm_mask]
        y_train_norm = y[idx_norm_mask] * 0


        print('X_train_norm, X_test_norm, y_train_norm, y_test_norm======', X_train_norm.shape,
              y_train_norm.shape, )

        #
        X_train_out = X[idx_out_mask]
        y_train_out = y[idx_out_mask] * 0 + 1


        X_train = np.concatenate((X_train_norm, X_train_out))
        X_test = X#np.concatenate((X_test_norm, X_test_out))
        y_train = np.concatenate((y_train_norm, y_train_out))
        y_train_semi = np.concatenate((y_train_norm, y_train_out * (-1)))
        y_test = y#np.concatenate((y_test_norm, y_test_out))
        y_test_semi = y * (-1)
        # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
        scaler = StandardScaler().fit(X_train)
        X_train_stand = scaler.transform(X_train)
        X_test_stand = scaler.transform(X_test)

        # Scale to range [0,1]
        minmax_scaler = MinMaxScaler().fit(X_train_stand)
        X_train_scaled = minmax_scaler.transform(X_train_stand)
        X_test_scaled = minmax_scaler.transform(X_test_stand)

        if self.train:
            self.data = torch.tensor(X_train_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.int64)
            self.semi_targets = torch.tensor(y_train_semi, dtype=torch.int64)
        else:
            self.data = torch.tensor(X_test_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.int64)
            self.semi_targets = torch.tensor(y_test_semi, dtype=torch.int64)

        # self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """
        sample, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])
        return sample, target, semi_target, index

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(self.data_file)
