from .mnist import MNIST_Dataset
from .fmnist import FashionMNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .odds import ODDSADDataset
from .HSI_1 import HSI_Dataset_1


def load_dataset(dataset_name, data_path, normal_class, known_outlier_class, cfg, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0,
                 random_state=None, patch_size=9):
    # dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, cfg, n_known_outlier_classes,
    #                            ratio_known_normal, ratio_known_outlier, ratio_pollution,
    #                            random_state=np.random.RandomState(cfg.settings['seed']),
    #                            patch_size=cfg.settings['patch_size'])
    """Loads the dataset."""

    implemented_datasets = ('HYDICE_urban_resize')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name in ('HYDICE_urban_resize'):
        dataset = ODDSADDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)

    return dataset
