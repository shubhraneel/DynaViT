from deit_modified_ghost import VisionTransformer
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import timm
import torch
from torch import nn, einsum
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, ImageFolder
from torchvision import transforms
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from utils import train_model, print_accuracy
from typing import Union
from sklearn.metrics import accuracy_score
from reorder_head_neuron import compute_neuron_head_importance, reorder_neuron_head

from functools import partial

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--test",
    action="store_true",
    help="To only test the model with the given weights file",
)
parser.add_argument(
    "--reorder",
    action="store_true",
    help="Load model, reorder and save it"
)
parser.add_argument(
    "--path_train",
    type=str,
    default="../data/ImageNet200FullSize/train",
    help="Path to train dataset (default: '../data/ImageNet200FullSize/train')",
)
parser.add_argument(
    "--path_val",
    type=str,
    default="../data/ImageNet200FullSize/val",
    help="Path to validation dataset (default: '../data/ImageNet200FullSize/val')",
)
parser.add_argument(
    "--model_architecture",
    type=str,
    default="vit_small_patch16_224",
    help="Architecture of model (default: 'vit_small_patch16_224')",
)
parser.add_argument(
    "--pretrained",
    action="store_true",
    help="To load pretrained weights of architecture instead of loading from model_path",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="../models/vit-small-224.pth",
    help="Path to model initial checkpoint OR to store state dictionary if pretrained is true (default: '../models/vit-small-224.pth')",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="../models/vit-small-224-finetuned-1.0.pth",
    help="Path to save model checkpoint OR for loading test model (default: '../models/vit-small-224-finetuned-1.0.pth')",
)
parser.add_argument(
    "--reorder_path",
    type=str,
    default="../models/vit-small-224-finetuned-1.0-reordered.pth",
    help="Path to save reordered model checkpoint (default: '../models/vit-small-224-finetuned-1.0-reordered.pth')",
)
parser.add_argument(
    "--mha_width",
    type=float,
    default=1.0,
    help="Width of multi head attention (default: 1.0)",
)
parser.add_argument(
    "--mlp_width",
    type=float,
    default=1.0,
    help="Width of feed forward layer (default: 1.0)",
)
parser.add_argument(
    "--epochs", type=int, default=50, help="Number of epochs (default: 50)"
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="Device to train (or test) on (default: 'cuda:0')",
)
parser.add_argument(
    "--batch_size", type=int, default=64, help="Batch size (default: 64)"
)
parser.add_argument(
    "--num_classes", type=int, default=1000, help="Number of classes (default: 1000)"
)
parser.add_argument(
    "--img_size",
    type=int,
    default=224,
    help="Image size (default: 224)",
)
parser.add_argument(
    "--patch_size", type=int, default=16, help="Patch size (default: 16)"
)
parser.add_argument(
    "--embed_dim", type=int, default=384, help="Embedding dimension (default: 384)"
)
parser.add_argument("--depth", type=int, default=12, help="Depth (default: 12)")
parser.add_argument(
    "--num_heads", type=int, default=6, help="Number of heads (default: 6)"
)
parser.add_argument("--mlp_ratio", type=int, default=4, help="MLP ratio (default: 4)")
parser.add_argument(
    "--in_chans", type=int, default=3, help="Input channels (default: 3)"
)
parser.add_argument("--qkv_bias", action="store_true", help="To use qkv bias")
parser.add_argument("--no_ghost", action="store_true", help="To not use ghost")
parser.add_argument(
    "--ghost_mode",
    type=str,
    default="simple",
    help="Mode for applying ghost module (default: 'simple')",
)
parser.add_argument("--init_scratch", action="store_true", help="To start from scratch model")

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device(args.device)
    print(f"{torch.cuda.device_count()} GPU(s) available.")
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

path_train = args.path_train
path_val = args.path_val

if args.pretrained == True:
    model_base = timm.create_model(args.model_architecture, pretrained=True)
    torch.save(model_base.state_dict(), args.model_path)

train_transforms = create_transform(
    input_size=args.img_size,
    is_training=True,
    color_jitter=0.4,
    auto_augment="rand-m9-mstd0.5-inc1",
    interpolation="bicubic",
    re_prob=0.25,
    re_mode="pixel",
    re_count=1,
)
val_transforms = transforms.Compose(
    [
        transforms.Resize(args.img_size + 32),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ]
)

train_dataset = ImageFolder(path_train, transform=train_transforms)
train_sampler = RandomSampler(train_dataset)
train_loader = DataLoader(
    train_dataset, sampler=train_sampler, batch_size=args.batch_size
)
val_dataset = ImageFolder(path_val, transform=val_transforms)
val_sampler = SequentialSampler(val_dataset)
test_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.batch_size)

model = VisionTransformer(
    img_size=args.img_size,
    patch_size=args.patch_size,
    num_classes=args.num_classes,
    embed_dim=args.embed_dim,
    depth=args.depth,
    num_heads=args.num_heads,
    mlp_ratio=args.mlp_ratio,
    in_chans=args.in_chans,
    qkv_bias=args.qkv_bias,
    mha_width=args.mha_width,
    mlp_width=args.mlp_width,
    no_ghost=args.no_ghost,
    ghost_mode=args.ghost_mode,
)

if args.test:
    model.load_state_dict(torch.load(args.save_path), strict=False)
    print_accuracy(model, test_loader, device=device)

    if args.reorder:
        head_importance = compute_neuron_head_importance(model, test_loader, num_layers=args.depth, num_heads=args.num_heads, device=device)
        reorder_neuron_head(model, head_importance)
        print("After reordering ... ")
        print_accuracy(model, test_loader, device=device)
        torch.save(model.state_dict(), args.reorder_path)

else:
    dim = args.embed_dim
    num_heads = args.num_heads
    if not args.init_scratch:
        path = args.model_path
        load_dict = torch.load(path)

        model_dict = model.state_dict()
        new_dict = {}
        head_dim = dim // num_heads

        for param in model_dict:
            if param in load_dict:
                if param == "head.weight":
                    new_dict["head.weight"] = torch.randn(args.num_classes, args.embed_dim).to(device)
                elif param == "head.bias":
                    new_dict["head.bias"] = torch.randn(args.num_classes).to(device)
                else:
                    new_dict[param] = load_dict[param].to(device)
            elif "attn.projs" in param:
                name_parts = param.split("projs.")
                name = name_parts[0] + "proj.weight"
                i = int(name_parts[1].split(".")[0])
                new_dict[param] = load_dict[name][:, head_dim * i : head_dim * (i + 1)].to(
                    device
                )
            elif param.endswith("attn.proj_bias"):
                name_parts = param.split("proj_bias")
                name = name_parts[0] + "proj.bias"
                new_dict[param] = load_dict[name].to(device)

        model.load_state_dict(new_dict, strict=False)
    optimizer = Adam(model.parameters(), lr=5e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.3)

    train_model(
        model,
        train_loader,
        test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        loss_fn=nn.CrossEntropyLoss(),
        path=args.save_path,
        device=device,
    )

    model.load_state_dict(torch.load(args.save_path))
    print_accuracy(model, test_loader, device=device)
