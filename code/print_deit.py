# -*- coding: utf-8 -*-
"""VisionTransformers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OtvGSH_lSOHjmnu_Dsepz2xj_FGnnA6n
"""

from timm.models.vision_transformer import VisionTransformer
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import torch
from torch import nn, einsum
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, ImageFolder
from torchvision import transforms
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from utils import count_parameters

from functools import partial

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'{torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
print(count_parameters(model))
