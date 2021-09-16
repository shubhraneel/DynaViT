from shutil import copyfile
import os

class_dict = {}

path = "./data/tiny-imagenet-200/val"

with open(os.path.join(path, "val_annotations.txt")) as f:
    for line in f:
        l = line.split()
        if l[1] not in class_dict:
            class_dict[l[1]] = [] 
        class_dict[l[1]].append(l[0])

for class_l in class_dict:
    p = os.path.join(path, class_l, "images")
    os.makedirs(p)
    for f in class_dict[class_l]:
        print(f"Copying file {os.path.join(path, 'images', f)} to {p}")
        copyfile(os.path.join(path, "images", f), p)
        
class_dict = {}

path = "./data/tiny-imagenet-200/test"

with open(os.path.join(path, "val_annotations.txt")) as f:
    for line in f:
        l = line.split()
        if l[1] not in class_dict:
            class_dict[l[1]] = [] 
        class_dict[l[1]].append(l[0])

for class_l in class_dict:
    p = os.path.join(path, class_l, "images")
    os.makedirs(p)
    for f in class_dict[class_l]:
        print(f"Copying file {os.path.join(path, 'images', f)} to {p}")
        copyfile(os.path.join(path, "images", f), p)
