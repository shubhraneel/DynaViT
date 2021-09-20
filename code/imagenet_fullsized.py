from shutil import copyfile
import os

src_path = "/scratch2/datasets/IMAGENET/"
alt_path = "/home/adaq/workspace/SlimmableViT/DynaViT/data/tiny-imagenet-200/"
dst_path = "/home/adaq/workspace/SlimmableViT/DynaViT/data/ImageNet200FullSize/"

for folder in os.listdir(os.path.join(alt_path, "train")):
    for f in os.listdir(os.path.join(src_path, "IMAGENET_TRAIN/", folder))[:500]:
        copyfile(os.path.join(src_path, "train/", folder, f))

for folder in os.listdir(os.path.join(alt_path, "val")):
    for f in os.listdir(os.path.join(src_path, "IMAGENET_VAL/", folder))[:500]:
        copyfile(os.path.join(src_path, "val/", folder, f))
