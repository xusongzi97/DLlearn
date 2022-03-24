import torch
import torchvision
import numpy
import matplotlib.pyplot as plt

train_set = torchvision.datasets.MNIST(root='../dataset/mnist', train=True, download=True)
# train_set:60000, test_set:10000, calsses:10
test_set = torchvision.datasets.MNIST(root='../dataste/mnist', train=False, download=True)
# 下载训练集和测试集

