from dynavision import DynamicVisionTransformer, train, print_metrics
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import torch
from torch import nn, einsum
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, ImageFolder
from torchvision import transforms
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm

from functools import partial

path_train = "../data/ImageNet200FullSize/train"
path_val = "../data/ImageNet200FullSize/val"

train_transforms = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1
        )
val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

train_dataset = ImageFolder(path_train, transform=train_transforms)
train_sampler = RandomSampler(train_dataset)
train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=64)
val_dataset = ImageFolder(path_val, transform=val_transforms)
val_sampler = SequentialSampler(val_dataset)
test_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=64)

train(train_loader, test_loader, mode="full", 
    model_path="../models/", epochs=100, img_size=224, patch_size=16, num_classes=200, 
    embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,  in_chans = 3, drop_rate=0, attn_drop_rate=0, drop_path_rate=0, distilled=False, act_layer=None, representation_size=None, loss_fn=nn.CrossEntropyLoss(),
    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
)

model = DynamicVisionTransformer(img_size=224, patch_size=16, 
        num_classes=200, 
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,  
        in_chans = 3, qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6))

model.load_state_dict(torch.load("../models/model_width_naive.pt"))

from sklearn.metrics import accuracy_score
print_metrics(model, test_loader, [
     (accuracy_score, {}), 
      ])

