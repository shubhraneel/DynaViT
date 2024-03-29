from dynavit import DynaViT, train, print_metrics

from torchvision.datasets import ImageFolder, CIFAR10
from torchvision import transforms
from sklearn.metrics import accuracy_score

from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
import torch
import torch.nn as nn
import sys

"""### CIFAR-10"""
path = "../data/"

train_dataset = CIFAR10(path, transform=transforms.ToTensor())
train_sampler = RandomSampler(train_dataset)
train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=64)
test_dataset = CIFAR10(path, transform=transforms.ToTensor(), train=False)
test_sampler = SequentialSampler(test_dataset)
test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=64)

train(
    train_loader, test_loader, mode="width", method='sandwich', width_list = [0.2, 0.4, 0.6, 0.8, 1],
    image_size=32, patch_size=8, num_classes=10, dim=256, dim_head=64, heads=4,
    depth=6, mlp_dim=512, dropout=0.1, emb_dropout=0.1, channels=3, pool='cls',
    epochs=40, loss_fn=nn.CrossEntropyLoss(), model_path="../models/cifar10"
    )
