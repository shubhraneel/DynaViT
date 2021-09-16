from dynavit import DynaViT, train, print_metrics

from torchvision.datasets import ImageNet
from torchvision import transforms
from sklearn.metrics import accuracy_score

from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
import torch
import torch.nn as nn
import sys

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'{torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    sys.exit('No GPU available.')
    

"""### CIFAR-10"""
path = "../data/tiny-imagenet-200"

train_dataset = ImageNet(path, transform=transforms.ToTensor())
train_sampler = RandomSampler(train_dataset)
train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=64)
test_dataset = ImageNet(path, split='val', transform=transforms.ToTensor(), train=False)
test_sampler = SequentialSampler(test_dataset)
test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=64)

train(
    train_loader, test_loader, mode='width', method='incremental', width_list = [0.2, 0.4, 0.6, 0.8, 1],
    image_size=32, patch_size=8, num_classes=10, dim=256, dim_head=64, heads=4,
    depth=6, mlp_dim=512, dropout=0.1, emb_dropout=0.1, channels=3, pool='cls',
    epochs=10, loss_fn=nn.CrossEntropyLoss(), model_path="../models/imagenet200"
    )

model = DynaViT(
            image_size=32, patch_size=8, num_classes=10, dim=256, dim_head=64, heads=4,
            depth=6, mlp_dim=512, dropout=0.1, emb_dropout=0.1, channels=3, pool='cls'
        )
model.load_state_dict(torch.load("../models/imagenet200"))

print_metrics(
    model, test_loader,
    [
     (accuracy_score, {}),
      ],
    )
