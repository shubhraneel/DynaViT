from dynavit import DynaViT, train, print_metrics

from torchvision.datasets import ImageFolder, CIFAR10
from torchvision import transforms
from sklearn.metrics import accuracy_score

from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
import torch
import torch.nn as nn
import sys

path = "../data/"

test_dataset = CIFAR10(path, transform=transforms.ToTensor(), train=False)
test_sampler = SequentialSampler(test_dataset)
test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=64)

model = DynaViT(
            image_size=32, patch_size=8, num_classes=10, dim=256, dim_head=64, heads=4,
            depth=6, mlp_dim=512, dropout=0.1, emb_dropout=0.1, channels=3, pool='cls'
        )
model.load_state_dict(torch.load("../models/cifar10/model_width_sandwich.pt"))

print_metrics(
    model, test_loader,
    [
     (accuracy_score, {}),
      ],
    width_list=[0.2, 0.4, 0.6, 0.8, 1.0]
    
    )
