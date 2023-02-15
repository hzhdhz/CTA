# Convolutional Transformer-inspired Autoencoder for Hyperspectral Anomaly Detection

This repository provides a [PyTorch](https://pytorch.org/) implementation of the *CTAnet* method presented in our paper
 ”Convolutional Transformer-inspired Autoencoder for Hyperspectral Anomaly Detection”.

how to run (train + test)?
HYDICE_urban
activate pytorch
python main.py HYDICE_urban_resize HYDICE_urban_CTA ./log/HYDICE_urban_CTA ./data --ratio_known_outlier 0.01 --ratio_pollution 0 --lr 0.0001 --n_epochs 100 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 100 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class 0 --known_outlier_class 1 --n_known_outlier_classes 1



The code is written based on the DSAD (https://github.com/lukasruff/Deep-SAD-PyTorch).

