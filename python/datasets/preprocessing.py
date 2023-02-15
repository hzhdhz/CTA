import torch
import numpy as np


def create_semisupervised_setting(labels, normal_classes, outlier_classes, known_outlier_classes,
                                  ratio_known_normal, ratio_known_outlier, ratio_pollution):
    # idx, _, semi_targets = create_semisupervised_setting(train_set.targets.cpu().data.numpy(), self.normal_classes,
    #                                                      self.outlier_classes, self.known_outlier_classes,
    #                                                      ratio_known_normal, ratio_known_outlier, ratio_pollution)
    """
    Create a semi-supervised data setting. 
    :param labels: np.array with labels of all dataset samples
    :param normal_classes: tuple with normal class labels
    :param outlier_classes: tuple with anomaly class labels
    :param known_outlier_classes: tuple with known (labeled) anomaly class labels
    :param ratio_known_normal: the desired ratio of known (labeled) normal samples
    :param ratio_known_outlier: the desired ratio of known (labeled) anomalous samples
    :param ratio_pollution: the desired pollution ratio of the unlabeled data with unknown (unlabeled) anomalies.
    :return: tuple with list of sample indices, list of original labels, and list of semi-supervised labels
    """
    idx_normal = np.argwhere(np.isin(labels, normal_classes)).flatten()
    idx_outlier = np.argwhere(np.isin(labels, outlier_classes)).flatten()
    idx_known_outlier_candidates = np.argwhere(np.isin(labels, known_outlier_classes)).flatten()

    n_normal = len(idx_normal)
    print('len(idx_normal), len(idx_outlier), len(idx_known_outlier_candidates)===================', normal_classes, outlier_classes, known_outlier_classes,
          len(idx_normal), len(idx_outlier), len(idx_known_outlier_candidates))
    #len(idx_normal), len(idx_outlier), len(idx_known_outlier_candidates)=================== (0,) (1,) (1,) 29409 73 73

    #len(idx_normal), len(idx_outlier), len(idx_known_outlier_candidates)=================== (0,) (1,) (1,) 4787 12 12

    #len(idx_normal), len(idx_outlier), len(idx_known_outlier_candidates)=================== (0,) (1,) (1,) 3853 7 7

    #len(idx_normal), len(idx_outlier), len(idx_known_outlier_candidates)=================== (0,) (1,) (1,) 2639 1221 1221
    #n_known_normal n_unlabeled_normal n_unlabeled_outlier n_known_outlier===== 0 2639 0 26
    #list_idx, list_labels, list_semi_labels===================== 2665 2665 2665


    # Solve system of linear equations to obtain respective number of samples
    a = np.array([[1, 1, 0, 0],
                  [(1-ratio_known_normal), -ratio_known_normal, -ratio_known_normal, -ratio_known_normal],
                  [-ratio_known_outlier, -ratio_known_outlier, -ratio_known_outlier, (1-ratio_known_outlier)],
                  [0, -ratio_pollution, (1-ratio_pollution), 0]])
    b = np.array([n_normal, 0, 0, 0])
    x = np.linalg.solve(a, b)

    # Get number of samples
    n_known_normal = int(x[0])
    n_unlabeled_normal = int(x[1])
    n_unlabeled_outlier = int(x[2])
    n_known_outlier = int(x[3])

    # Sample indices
    perm_normal = np.random.permutation(n_normal)
    perm_outlier = np.random.permutation(len(idx_outlier))
    perm_known_outlier = np.random.permutation(len(idx_known_outlier_candidates))

    idx_known_normal = idx_normal[perm_normal[:n_known_normal]].tolist()
    idx_unlabeled_normal = idx_normal[perm_normal[n_known_normal:n_known_normal+n_unlabeled_normal]].tolist()
    idx_unlabeled_outlier = idx_outlier[perm_outlier[:n_unlabeled_outlier]].tolist()
    idx_known_outlier = idx_known_outlier_candidates[perm_known_outlier[:n_known_outlier]].tolist()
    print('idx_known_outlier=================', len(idx_known_outlier))
    #idx_known_outlier================= 12

    # Get original class labels
    labels_known_normal = labels[idx_known_normal].tolist()
    labels_unlabeled_normal = labels[idx_unlabeled_normal].tolist()
    labels_unlabeled_outlier = labels[idx_unlabeled_outlier].tolist()
    labels_known_outlier = labels[idx_known_outlier].tolist()

    # Get semi-supervised setting labels
    # n_known_normal = int(x[0])
    # n_unlabeled_normal = int(x[1])
    # n_unlabeled_outlier = int(x[2])
    # n_known_outlier = int(x[3])
    semi_labels_known_normal = np.ones(n_known_normal).astype(np.int32).tolist()
    semi_labels_unlabeled_normal = np.zeros(n_unlabeled_normal).astype(np.int32).tolist()
    semi_labels_unlabeled_outlier = np.zeros(n_unlabeled_outlier).astype(np.int32).tolist()
    # semi_labels_known_outlier = (-np.ones(n_known_outlier).astype(np.int32)).tolist()
    semi_labels_known_outlier = (-np.ones(len(idx_known_outlier)).astype(np.int32)).tolist()

    # Create final lists
    list_idx = idx_known_normal + idx_unlabeled_normal + idx_unlabeled_outlier + idx_known_outlier
    list_labels = labels_known_normal + labels_unlabeled_normal + labels_unlabeled_outlier + labels_known_outlier
    list_semi_labels = (semi_labels_known_normal + semi_labels_unlabeled_normal + semi_labels_unlabeled_outlier
                        + semi_labels_known_outlier)
    # n_known_normal = int(x[0])
    # n_unlabeled_normal = int(x[1])
    # n_unlabeled_outlier = int(x[2])
    # n_known_outlier = int(x[3])
    print('n_known_normal n_unlabeled_normal n_unlabeled_outlier n_known_outlier=====', n_known_normal, n_unlabeled_normal, n_unlabeled_outlier, n_known_outlier)
    #n_known_normal n_unlabeled_normal n_unlabeled_outlier n_known_outlier===== 0 4787 0 48
    # n_known_normal n_unlabeled_normal n_unlabeled_outlier n_known_outlier===== 0 3853 0 38
    #list_idx, list_labels, list_semi_labels===================== 4799 4799 4835

    #n_known_normal n_unlabeled_normal n_unlabeled_outlier n_known_outlier===== 0 2639 0 26

    print('list_idx, list_labels, list_semi_labels=====================', len(list_idx), len(list_labels), len(list_semi_labels))
    #list_idx, list_labels, list_semi_labels===================== 4799 4799 5318


    #list_idx, list_labels, list_semi_labels===================== 2665 2665 2665

    # print('list_semi_labels==========================', list_idx)
    #list_semi_labels========================== [3175, 26371, 41609, 53537,
    # print('list_labels==========================', list_labels)
    #[ 0, 0, 5, 9, 4, 4, 7, 4, 5, 7, 4, 7, 7, 8, 6, 9, 3, 6, 3, 9, 7, 6, 6, 9, 8, 1, 9, 4, 2, 4, 9, 9, 1, 9, 4, 9, 6, 2, 2, 4, 7, 1,
# 2, 1, 2, 7, 9, 4, 5, 2, 3, 3, 2, 4, 1, 5, 9, 3, 7, 4, 8, 2,]
    # print('list_semi_labels==========================', list_semi_labels)
    #list_idx, list_labels, list_semi_labels===================== 6647 6647 6647
    #list_semi_labels========================== [0, -1, -1, -1, -1,]
    #idx, _, semi_targets

    return list_idx, list_labels, list_semi_labels
