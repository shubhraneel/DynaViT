from dynavit import DynaViT, train, print_metrics

from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import accuracy_score

from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
import torch
import torch.nn as nn
import sys

path_val = "../data/tiny-imagenet-200/val"
val_dataset = ImageFolder(path_val, transform=transforms.ToTensor())
val_sampler = SequentialSampler(val_dataset)
val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=64)

model = DynaViT(
            image_size=64, patch_size=16, num_classes=200, dim=512, dim_head=64, heads=8,
            depth=6, mlp_dim=1024, dropout=0.1, emb_dropout=0.1, channels=3, pool='cls',
        )
model.load_state_dict(torch.load("../models/imagenet200/model.pt"))

print_metrics(
    model, val_loader,
    [
     (accuracy_score, {}),
      ],
    )
