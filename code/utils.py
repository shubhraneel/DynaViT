"""### Importing Libraries"""

import torch
from torch import nn, einsum
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from torch.optim import Adam, lr_scheduler
from torchvision import transforms

#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pickle
import os
import re

from timm.models.layers import DropPath, PatchEmbed, trunc_normal_, lecun_normal_
from deit_modified import VisionTransformer

from functools import partial

from tqdm import tqdm

#try:
    #from google.colab import drive
    #drive.mount("/content/gdrive")
    #colab = True
#except:
    #colab = False

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'{torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#helpers
#def show_torch_img(image):
    #plt.imshow(transforms.ToPILImage()(image))

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def soft_cross_entropy(predicts, targets):
    student_likelihood = nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = nn.functional.softmax(targets, dim=-1)
    return -torch.sum(targets_prob * student_likelihood, dim=-1).mean()


"""### Importance reordering"""

def compute_neuron_head_importance(
    eval_dataloader, model, n_layers, n_heads, loss_fn=nn.CrossEntropyLoss()
    ):
    """ This method shows how to compute:
        - neuron importance scores based on loss according to http://arxiv.org/abs/1905.10650
    """
    
    head_importance = torch.zeros(n_layers, n_heads).to(device)
    head_mask = torch.ones(n_layers, n_heads).to(device)
    head_mask.requires_grad_(requires_grad=True)

    intermediate_weight = []
    intermediate_bias = []
    output_weight = []
    for name, w in model.named_parameters():
        if 'intermediate' in name:
            if w.dim() > 1:
                intermediate_weight.append(w)
            else:
                intermediate_bias.append(w)

        if 'output' in name and 'attention' not in name:
            if w.dim() > 1:
                output_weight.append(w)
    
    neuron_importance = []
    for w in intermediate_weight:
        neuron_importance.append(torch.zeros(w.shape[0]).to(device))
    
    model.to(device)

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, label_ids = batch

        # calculate head importance
        outputs = model(input_ids, head_mask=head_mask)
        loss = loss_fn(outputs, label_ids)
        loss.backward()
        head_importance += head_mask.grad.abs().detach()

        # calculate  neuron importance
        for w1, b1, w2, current_importance in zip(intermediate_weight, intermediate_bias, output_weight, neuron_importance):
            current_importance += ((w1 * w1.grad).sum(dim=1) + b1 * b1.grad).abs().detach()
            current_importance += ((w2 * w2.grad).sum(dim=0)).abs().detach()
    
    return head_importance, neuron_importance

def reorder_neuron_head(model, head_importance, neuron_importance):

    model = model.module if hasattr(model, 'module') else model

    # reorder heads and ffn neurons
    for layer, current_importance in enumerate(neuron_importance):
        # reorder heads --> [layer][0]: attention module
        idx = torch.sort(head_importance[layer], descending=True)[-1]
        model.transformer.layers[layer][0].fn.reorder_heads(idx)
        # reorder neurons --> [layer][1]: feed-forward module
        idx = torch.sort(current_importance, descending=True)[-1]
        model.transformer.layers[layer][1].fn.reorder_intermediate_neurons(idx)
        model.transformer.layers[layer][1].fn.reorder_output_neurons(idx)

"""### Training"""

def train(
    model, train_data, eval_data, optimizer, scheduler, 
    mode = "full", method = None, width_list = None, width_sep = None,
    weights_file = None, model_path = "./", loss_fn=nn.CrossEntropyLoss(), epochs=100, prefix=""
    ):
    assert mode in ["full", "width", "height"], "Wrong mode input"

    # if "model" not in args:
    #     print("No pretrained model found, initializing model")
    #     if method=="difflayernorm":
    #         model = VisionTransformer(
    #                     img_size=args["img_size"], patch_size=args["patch_size"], 
    #                     num_classes=args["num_classes"], embed_dim=args["embed_dim"], 
    #                     depth=args["depth"], num_heads=args["num_heads"], mlp_ratio=args["mlp_ratio"], 
    #                     drop_rate = args["drop_rate"],
    #                     attn_drop_rate = args["attn_drop_rate"],
    #                     drop_path_rate = args["drop_path_rate"],
    #                     distilled = args["distilled"],
    #                     qkv_bias = args["qkv_bias"],
    #                     norm_layer = args["norm_layer"],
    #                     act_layer = args["act_layer"],
    #                     representation_size = args["representation_size"],
    #                     width_mult = 1.0, widths=len(width_list)
    #         )
    #     else:
    #         model = VisionTransformer(
    #                     img_size=args["img_size"], patch_size=args["patch_size"], 
    #                     num_classes=args["num_classes"], embed_dim=args["embed_dim"], 
    #                     depth=args["depth"], num_heads=args["num_heads"], mlp_ratio=args["mlp_ratio"], 
    #                     drop_rate = args["drop_rate"],
    #                     attn_drop_rate = args["attn_drop_rate"],
    #                     drop_path_rate = args["drop_path_rate"],
    #                     distilled = args["distilled"],
    #                     qkv_bias = args["qkv_bias"],
    #                     norm_layer = args["norm_layer"],
    #                     act_layer = args["act_layer"],
    #                     representation_size = args["representation_size"],
    #                     width_mult = 1.0
    #         )
    # else:
        # args["model_path"] = os.path.join(args["model_path"], "retrained")

    model.to(device)

    if weights_file is not None:
        model.load_state_dict(torch.load(weights_file))
    
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    if mode == "full":
        train_model(model, path = os.path.join(model_path, f"{prefix}model.pt"),
                    train_data = train_data, eval_data = eval_data,
                    epochs = epochs, loss_fn = loss_fn,
                    optimizer=optimizer, scheduler=scheduler
                    )
    
    if mode == "width":

        if method == "separate":
            print(f"Training separately {width_sep}")
            model.apply(lambda m: setattr(m, 'width_mult', width_sep))
            path = os.path.join(model_path, f"{prefix}model_width_separate_{width_sep}.pt")
            train_model(
                model, path = path,
                train_data = train_data, eval_data = eval_data,
                epochs = epochs, loss_fn = loss_fn,
                optimizer=optimizer, scheduler=scheduler
                )

        if method == "sandwich":
            print("Training Sandwich")
            width_list = sorted(width_list)
            path = os.path.join(model_path, f"{prefix}model_width_sandwich.pt")
            train_sandwich(
                    model, path = path,
                    train_data = train_data, eval_data = eval_data,
                    epochs = epochs, loss_fn = loss_fn,
                    width_min = width_list[0], width_max = width_list[-1], n_widths=len(width_list), 
                    optimizer=optimizer, scheduler=scheduler
                    )
        
        if method == "naive":
            print("Training Naive")
            width_list = sorted(width_list)
            path = os.path.join(model_path, f"{prefix}model_width_naive.pt")
            train_naive(
                    model, path = path,
                    train_data = train_data, eval_data = eval_data,
                    epochs = epochs, loss_fn = loss_fn,
                    width_list = width_list, optimizer=optimizer, scheduler=scheduler
                    )
            
        if method == "difflayernorm":
            print("Training Different Layer Norm wise")
            width_list = sorted(width_list)
            path = os.path.join(args["model_path"], "model_width_naive.pt")
            train_naive(
                    model, path = path,
                    train_data = train_data, eval_data = eval_data,
                    epochs = args["epochs"], loss_fn = args["loss_fn"],
                    width_list = width_list, optimizer=optimizer, scheduler=scheduler,
                    layernorm = True
                    )
        
        if method == "incremental":
            print("Training Incremental")
            width_list = sorted(width_list)
            path = os.path.join(args["model_path"], "model_width_incremental.pt")
            for i, width in enumerate(tqdm(width_list, desc="Width", leave=False)):
                print(f"\nWidth: {width}")
                model.apply(lambda m: setattr(m, 'width_mult', width))
                train_incremental(
                    model, path = path,
                    train_data = train_data, eval_data = eval_data,
                    epochs = args["epochs"], loss_fn = args["loss_fn"],
                    optimizer=optimizer, scheduler=scheduler,
                    freeze_width = width_list[i-1] if i > 0 else None
                    )
                model.load_state_dict(torch.load(path))

        if method == "distillation":
            print("Training Distillation")
            teacher_model = DynaViT(
                image_size=args["image_size"], patch_size=args["patch_size"], 
                num_classes=args["num_classes"], dim=args["dim"], 
                depth=args["depth"], heads=args["heads"], mlp_dim=args["mlp_dim"], 
                pool = args["pool"], channels = args["channels"], dim_head = args["dim_head"], 
                dropout = args["dropout"], emb_dropout = args["emb_dropout"],
                width_mult = 1.0
            )
            teacher_model.load_state_dict(torch.load(args["teacher_weights"]))
            head_importance, neuron_importance = compute_neuron_head_importance(
                eval_data, model, args["depth"], args["heads"], 
                loss_fn=args["loss_fn"]
                )
            reorder_neuron_head(teacher_model, head_importance, neuron_importance)
            width_list = sorted(width_list, reverse=True)
            path = os.path.join(args["model_path"], "model_width_distillation.pt")
            for i, width in enumerate(tqdm(width_list, desc="Width", leave=False)):
                print(f"\nWidth: {width}")
                model.apply(lambda m: setattr(m, 'width_mult', width))
                train_distillation(
                    model, teacher_model = teacher_model, path = path,
                    train_data = train_data, eval_data = eval_data,
                    epochs = args["epochs"],
                    optimizer=optimizer, scheduler=scheduler,
                    lambda1 = args["lambda1"], lambda2 = args["lambda2"]
                )
                model.load_state_dict(torch.load(path))
            print("Fine tuning after distillation")
            path = os.path.join(args["model_path"], "model_width_distillation_finetuned.pt")
            for i, width in enumerate(tqdm(width_list, desc="Width", leave=False)):
                print(f"\nWidth: {width}")
                model.apply(lambda m: setattr(m, 'width_mult', width))
                train_model(
                    model, path = path,
                    train_data = train_data, eval_data = eval_data,
                    epochs = args["epochs"], loss_fn = args["loss_fn"],
                    optimizer=optimizer, scheduler=scheduler,
                    lambda1 = args["lambda1"], lambda2 = args["lambda2"]
                )
                model.load_state_dict(torch.load(path))

def train_model(model, train_data, eval_data, path, epochs, loss_fn, optimizer, scheduler, **args):
    model.train()
    best_eval_loss = 1e8
    
    for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
        total_loss = 0.0
        print(f"\nEpoch: {epoch}")

        for i, data in enumerate(tqdm(train_data, desc="Training", leave=False)):
            
            inputs, labels = tuple(t.to(device) for t in data)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()*inputs.size(0)
        
        print(f"Train loss = {total_loss/len(train_data.sampler)}")

        model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_data, desc="Evaluating", leave=False)):
                inputs, labels = tuple(t.to(device) for t in data)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                eval_loss += loss.item()*inputs.size(0)
        
        eval_loss = eval_loss/len(eval_data.sampler)
        scheduler.step(metrics=eval_loss)

        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), path)
            best_eval_loss = eval_loss

        print(f"Validation loss = {eval_loss}")

def train_naive(model, train_data, eval_data, path, epochs, loss_fn, 
                        width_list, optimizer, scheduler, layernorm=False, **args):
    model.train()
    best_eval_loss = 1e8
    
    for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
        total_loss = 0.0
        print(f"\nEpoch: {epoch}")

        for i, data in enumerate(tqdm(train_data, desc="Training", leave=False)):

            inputs, labels = tuple(t.to(device) for t in data)
            optimizer.zero_grad()
            width_list_loss = 0.0
            for j, width in enumerate(width_list):
                model.apply(lambda m: setattr(m, 'width_mult', width))
                if layernorm:
                    outputs = model(inputs, width_n=j)
                else:
                    outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                width_list_loss += loss.item()
                loss.backward()
            optimizer.step()

            total_loss += (width_list_loss/len(width_list))*inputs.size(0)
        
        print(f"Train loss = {total_loss/len(train_data.sampler)}")

        model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_data, desc="Evaluating", leave=False)):
                inputs, labels = tuple(t.to(device) for t in data)
                width_list_loss = 0.0
                for j, width in enumerate(width_list):
                    model.apply(lambda m: setattr(m, 'width_mult', width))
                    if layernorm:
                        outputs = model(inputs, width_n=j)
                    else:
                        outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    width_list_loss += loss.item()

                eval_loss += (width_list_loss/len(width_list))*inputs.size(0)
        
        eval_loss = eval_loss/len(eval_data.sampler)
        scheduler.step(metrics=eval_loss)

        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), path)
            best_eval_loss = eval_loss

        print(f"Validation loss = {eval_loss}")

def train_sandwich(model, train_data, eval_data, path, epochs, loss_fn,  
                        optimizer, scheduler, layernorm=False, width_min = 0.25, width_max = 1, n_widths=5, 
                   **args
                   ):
    model.train()
    best_eval_loss = 1e8
    
    for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
        total_loss = 0.0
        print(f"\nEpoch: {epoch}")

        for i, data in enumerate(tqdm(train_data, desc="Training", leave=False)):

            inputs, labels = tuple(t.to(device) for t in data)
            optimizer.zero_grad()
            width_list_loss = 0.0
            width_list = list(np.random.choice(np.arange(256*width_min, 256*width_max), n_widths-2))
            width_list = [width_min] + width_list + [width_max]
            for j, width in enumerate(width_list):
                model.apply(lambda m: setattr(m, 'width_mult', width))
                if layernorm:
                    outputs = model(inputs, width_n=j)
                else:
                    outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                width_list_loss += loss.item()
                loss.backward()
            optimizer.step()

            total_loss += (width_list_loss/len(width_list))*inputs.size(0)
        
        print(f"Train loss = {total_loss/len(train_data.sampler)}")

        model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_data, desc="Evaluating", leave=False)):
                inputs, labels = tuple(t.to(device) for t in data)
                width_list_loss = 0.0
                for j, width in enumerate(width_list):
                    model.apply(lambda m: setattr(m, 'width_mult', width))
                    if layernorm:
                        outputs = model(inputs, width_n=j)
                    else:
                        outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    width_list_loss += loss.item()

                eval_loss += (width_list_loss/len(width_list))*inputs.size(0)
        
        eval_loss = eval_loss/len(eval_data.sampler)
        scheduler.step(metrics=eval_loss)

        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), path)
            best_eval_loss = eval_loss

        print(f"Validation loss = {eval_loss}")

def train_incremental(model, train_data, eval_data, path, 
                      epochs, loss_fn, optimizer, scheduler, 
                      freeze_width=None, **args):
    model.train()
    best_eval_loss = 1e8

    def zero_grad_dyna_linear(x, width):
        if isinstance(x, DynaLinear):
            x.set_grad_to_zero(width)
    
    for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
        total_loss = 0.0
        print(f"\nEpoch: {epoch}")

        for i, data in enumerate(tqdm(train_data, desc="Training", leave=False)):

            inputs, labels = tuple(t.to(device) for t in data)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            
            # important step to not destroy previously learned parameters
            if freeze_width:
                model.apply(lambda x: zero_grad_dyna_linear(x, freeze_width))

            optimizer.step()

            total_loss += loss.item()*inputs.size(0)

        print(f"Train loss = {total_loss/len(train_data.sampler)}")

        model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_data, desc="Evaluating", leave=False)):
                inputs, labels = tuple(t.to(device) for t in data)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                eval_loss += loss.item()*inputs.size(0)

        eval_loss = eval_loss/len(eval_data.sampler)
        scheduler.step(metrics=eval_loss)

        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), path)
            best_eval_loss = eval_loss

        print(f"Validation loss = {eval_loss}")

def train_distillation(model, teacher_model, train_data, eval_data, path, 
                      epochs, optimizer, scheduler, lambda1, lambda2,
                      **args):
    model.train()
    best_eval_loss = 1e8

    loss_mse = nn.MSELoss()

    def zero_grad_dyna_linear(x, width):
        if isinstance(x, DynaLinear):
            x.set_grad_to_zero(width)
    
    for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
        total_loss = 0.0
        print(f"\nEpoch: {epoch}")

        for i, data in enumerate(tqdm(train_data, desc="Training", leave=False)):

            inputs, labels = tuple(t.to(device) for t in data)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_out, teacher_hidden = teacher_model(inputs, return_states=True)
            student_out, student_hidden = model(inputs, return_states=True)
            loss1 = soft_cross_entropy_loss(student_out, teacher_out.detach())
            loss2 = loss_mse(student_hidden, teacher_hidden.detach())

            loss = loss1*lambda1 + loss2*lambda2
            loss.backward()

            optimizer.step()

            total_loss += loss.item()*inputs.size(0)

        print(f"Train loss = {total_loss/len(train_data.sampler)}")

        model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_data, desc="Evaluating", leave=False)):
                inputs, labels = tuple(t.to(device) for t in data)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                eval_loss += loss.item()*inputs.size(0)

        eval_loss = eval_loss/len(eval_data.sampler)
        scheduler.step(metrics=eval_loss)

        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), path)
            best_eval_loss = eval_loss

        print(f"Validation loss = {eval_loss}")

def print_metrics(model, test_data, metric_funcs, loss_fn=None, width_list=None, width_switch=False):
    model.eval()
    model.to(device)

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    if width_list:
        print(f"Width | {'Loss':^20}", end = "")
        for metric, args in metric_funcs:
            print(f" | {metric.__name__:^20}", end = "")
        print()
        for k, width in enumerate(width_list):
            print(f"{width:^5}", end = "")
            model.apply(lambda m: setattr(m, 'width_mult', width))
            preds = []
            truths = []
            
            total_loss = 0
            with torch.no_grad():
                for i, data in enumerate(test_data):
                    inputs, labels = tuple(t.to(device) for t in data)
                    if width_switch:
                        outputs = model(inputs, width_n=k)
                    else:
                        outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    total_loss += loss.item()*inputs.size(0)
                    preds = preds + list(
                        torch.argmax(
                            nn.functional.softmax(outputs.cpu(), dim=1), 
                            dim=1
                            )
                        )
                    truths = truths + list(labels.cpu())
            test_loss = total_loss/len(test_data.sampler)
            print(f" | {test_loss:^20.4f}", end = "")
            for metric, args in metric_funcs:
                perf = metric(truths, preds, **args)
                print(f" | {perf:^20.4f}", end = "")
            print()
    else:
        preds = []
        truths = []
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_data):
                inputs, labels = tuple(t.to(device) for t in data)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()*inputs.size(0)
                preds = preds + list(
                    torch.argmax(
                        nn.functional.softmax(outputs.cpu(), dim=1), 
                        dim=1
                        )
                    )
                truths = truths + list(labels.cpu())
        test_loss = total_loss/len(test_data.sampler)
        print(f"Loss: {test_loss}")
        for metric, args  in metric_funcs:
            perf = metric(truths, preds, **args)
            print(f"{metric.__name__}: {perf:^.4f}")

def print_accuracy(model, test_data, loss_fn=None, width_list=None, width_switch=False):
    model.eval()
    model.to(device)
    metric_funcs = [(accuracy_score, {})]

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    if width_list:
        print(f"Width | {'Loss':^20}", end = "")
        for metric, args in metric_funcs:
            print(f" | {metric.__name__:^20}", end = "")
        print()
        for k, width in enumerate(width_list):
            print(f"{width:^5}", end = "")
            model.apply(lambda m: setattr(m, 'width_mult', width))
            preds = []
            truths = []
            
            total_loss = 0
            with torch.no_grad():
                for i, data in enumerate(test_data):
                    inputs, labels = tuple(t.to(device) for t in data)
                    if width_switch:
                        outputs = model(inputs, width_n=k)
                    else:
                        outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    total_loss += loss.item()*inputs.size(0)
                    preds = preds + list(
                        torch.argmax(
                            nn.functional.softmax(outputs.cpu(), dim=1), 
                            dim=1
                            )
                        )
                    truths = truths + list(labels.cpu())
            test_loss = total_loss/len(test_data.sampler)
            print(f" | {test_loss:^20.4f}", end = "")
            for metric, args in metric_funcs:
                perf = metric(truths, preds, **args)
                print(f" | {perf:^20.4f}", end = "")
            print()
    else:
        preds = []
        truths = []
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_data):
                inputs, labels = tuple(t.to(device) for t in data)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()*inputs.size(0)
                preds = preds + list(
                    torch.argmax(
                        nn.functional.softmax(outputs.cpu(), dim=1), 
                        dim=1
                        )
                    )
                truths = truths + list(labels.cpu())
        test_loss = total_loss/len(test_data.sampler)
        print(f"Loss: {test_loss}")
        for metric, args  in metric_funcs:
            perf = metric(truths, preds, **args)
            print(f"{metric.__name__}: {perf:^.4f}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


