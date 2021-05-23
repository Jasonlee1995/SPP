# SPP-ZFNet Implementation with Pytorch
- Unofficial implementation of the paper *Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition*


## 0. Develop Environment
```
Docker Image
- pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
```
- Using Single GPU


## 1. Implementation Details
- model.py : ZFNet model with SPP
- train.py : train ZFNet (include 10-crop on val/test)
- utils.py : count correct prediction
- ZFNet - Cifar 10 (No SPP).ipynb : install library, download dataset, preprocessing, train and result of ZFNet model without SPP
- ZFNet - Cifar 10 (SPP).ipynb : install library, download dataset, preprocessing, train and result of ZFNet model with SPP
- Details
  * Follow ZFNet train details : batch size 128, learning rate 0.01, momentum 0.9, weight decay 0.0005
  * No learning rate scheduler for convenience


## 2. Reference
- Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition [[paper]](https://arxiv.org/abs/1406.4729)
