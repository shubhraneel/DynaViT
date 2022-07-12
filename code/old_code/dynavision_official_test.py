from deit_modified import VisionTransformer
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import torch
from torch import nn, einsum
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, ImageFolder
from torchvision import transforms
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from utils import train, print_accuracy

from functools import partial

path_val = "../data/ImageNet200FullSize/val"

val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

val_dataset = ImageFolder(path_val, transform=val_transforms)
val_sampler = SequentialSampler(val_dataset)
test_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=64)

model = VisionTransformer(img_size=224, patch_size=16, 
        num_classes=200, 
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,  
        in_chans = 3, qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6))

model.load_state_dict(torch.load("../models/officialmodel.pt"))

from sklearn.metrics import accuracy_score
print_accuracy(model, test_loader)