# DISH

This repository is the implementation of [***Scalable Discrete Supervised Hash Learning with Asymmetric Matrix Factorization***](https://arxiv.org/abs/1609.08740) (Full paper version, the short paper version is to appear on **ICDM'16**).

## Requirements

* numpy
* scipy
* caffe

## Running the code

1. Download the cifar-10 data(origin image in RGB format and GIST descriptor) in `npy` format and extract them to the root folder ([***link***](ml.cs.tsinghua.edu.cn/~shifeng/cifar10_data.tar.gz))
* Run `python dish-d.py` (for deep hashing) or `python dish-k.py` (for kernel-based hashing).
* NOTE: you may run `python resize_256.py` and download the weights of VGG-16 net before using VGG net for training
