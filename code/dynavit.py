# -*- coding: utf-8 -*-
"""DynaViT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OlHUmZxRsbG_7RDCHzI43_ZUtVDNIs6G
"""


"""### Importing Libraries"""

import torch
from torch import nn, einsum
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.optim import Adam, lr_scheduler
from torchvision import transforms

import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pickle
import os
import re

from tqdm import tqdm

device = torch.device('cuda:1')

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def soft_cross_entropy(predicts, targets):
    student_likelihood = nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = nn.functional.softmax(targets, dim=-1)
    return -torch.sum(targets_prob * student_likelihood, dim=-1).mean()

"""## Model Code

### Layer Norm
"""

class PreNorm(nn.Module):
    def __init__(self, dim, fn, widths = None):
        super().__init__()
        self.widths = widths
        if widths is None:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(widths)])
        self.fn = fn
    def forward(self, x, width_n=None, **kwargs):
        if self.widths is None:
            return self.fn(self.norm(x), **kwargs)
        return self.fn(self.norms[width_n](x), **kwargs)

"""### Feed Forward"""

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        # first layer of feed forward
        self.intermediate = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        # second layer of feed forward
        self.dropout_intermediate = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, dim)
        self.dropout_output = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.intermediate(x)
        x = self.activation(x)
        x = self.dropout_intermediate(x)
        x = self.output(x)
        x = self.dropout_output(x)
        return x

class DynamicFeedForward(FeedForward):
    def __init__(self, dim, intermediate_dim, num_heads, dropout=0., width_mult=1.0):
        super(DynamicFeedForward, self).__init__(dim, intermediate_dim, dropout)
        self.intermediate = DynaLinear(dim, intermediate_dim, num_heads, dyna_dim=[False, True], width_mult = width_mult)
        self.output = DynaLinear(intermediate_dim, dim, num_heads, dyna_dim=[True, False], width_mult = width_mult)
    
    def reorder_intermediate_neurons(self, index, dim=0):
        index = index.to(self.intermediate.weight.device)
        W = self.intermediate.weight.index_select(dim, index).clone().detach()
        if self.intermediate.bias is not None:
            if dim == 1:
                b = self.intermediate.bias.clone().detach()
            else:
                b = self.intermediate.bias[index].clone().detach()
        self.intermediate.weight.requires_grad = False
        self.intermediate.weight.copy_(W.contiguous())
        self.intermediate.weight.requires_grad = True
        if self.intermediate.bias is not None:
            self.intermediate.bias.requires_grad = False
            self.intermediate.bias.copy_(b.contiguous())
            self.intermediate.bias.requires_grad = True

    def reorder_output_neurons(self, index, dim=1):
        index = index.to(self.output.weight.device)
        W = self.output.weight.index_select(dim, index).clone().detach()
        if self.output.bias is not None:
            if dim == 1:
                b = self.output.bias.clone().detach()
            else:
                b = self.output.bias[index].clone().detach()
        self.output.weight.requires_grad = False
        self.output.weight.copy_(W.contiguous())
        self.output.weight.requires_grad = True
        if self.output.bias is not None:
            self.output.bias.requires_grad = False
            self.output.bias.copy_(b.contiguous())
            self.output.bias.requires_grad = True

"""### Multi-head self attention"""

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head

        self.attend = nn.Softmax(dim = -1)
        self.q = nn.Linear(dim, inner_dim, bias = False)
        self.k = nn.Linear(dim, inner_dim, bias = False)
        self.v = nn.Linear(dim, inner_dim, bias = False)

        if self.project_out:
            self.to_out = nn.Linear(inner_dim, dim)
            self.dropout = nn.Dropout(dropout)
        else:
            self.to_out = nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = [self.q(x), self.k(x), self.v(x)]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        if self.project_out:
            out = self.to_out(out)
            out = self.dropout(out)
        else:
            out = self.to_out(out)
        return out

"""### Multi-head Attention with variable width"""

class DynamicAttention(Attention):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., width_mult = 1.0):
        super(DynamicAttention, self).__init__(dim, heads, dim_head, dropout=dropout)
        inner_dim = heads * dim_head

        self.q = DynaLinear(dim, inner_dim, heads, dyna_dim=[False, True], width_mult=width_mult)
        self.k = DynaLinear(dim, inner_dim, heads, dyna_dim=[False, True], width_mult=width_mult)
        self.v = DynaLinear(dim, inner_dim, heads, dyna_dim=[False, True], width_mult=width_mult)

        if self.project_out:
            self.to_out = DynaLinear(inner_dim, dim, heads, dyna_dim=[True, False], width_mult=width_mult)
            self.dropout = nn.Dropout(dropout)
        else:
            self.to_out = nn.Identity()
    
    def reorder_heads(self, idx):
        n, a = self.heads, self.dim_head
        index = torch.arange(n*a).reshape(n, a)[idx].view(-1).contiguous().long()

        def reorder_head_matrix(linearLayer, index, dim=0):
            index = index.to(linearLayer.weight.device)
            W = linearLayer.weight.index_select(dim, index).clone().detach()
            if linearLayer.bias is not None:
                if dim == 1:
                    b = linearLayer.bias.clone().detach()
                else:
                    b = linearLayer.bias[index].clone().detach()

            linearLayer.weight.requires_grad = False
            linearLayer.weight.copy_(W.contiguous())
            linearLayer.weight.requires_grad = True
            if linearLayer.bias is not None:
                linearLayer.bias.requires_grad = False
                linearLayer.bias.copy_(b.contiguous())
                linearLayer.bias.requires_grad = True

        reorder_head_matrix(self.q, index)
        reorder_head_matrix(self.k, index)
        reorder_head_matrix(self.v, index)
        reorder_head_matrix(self.to_out, index, dim=1)
    
    def forward(self, x, head_mask = None):
        b, n, _, h = *x.shape, round(self.heads * self.q.width_mult)
        qkv = [self.q(x), self.k(x), self.v(x)]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        if head_mask is not None:
            head_mask = head_mask[:, :h, :, :]
            attn = attn * head_mask

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        if self.project_out:
            out = self.to_out(out)
            out = self.dropout(out)
        else:
            out = self.to_out(out)
        return out

"""### Encoder Layers"""

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, head_mask = None, return_states = False, width_n=None):
        hidden_states = []
        for i, (attn, ff) in enumerate(self.layers):
            x = attn(x, head_mask = head_mask[i], width_n=width_n) + x
            x = ff(x, width_n=width_n) + x
            hidden_states.append(x)
        if return_states:
            return x, hidden_states
        return x

class DynamicTransformer(Transformer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., width_mult = 1.0, widths=None):
        super(DynamicTransformer, self).__init__(dim, depth, heads, dim_head, mlp_dim, dropout=dropout)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, DynamicAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, width_mult = width_mult), widths=widths),
                PreNorm(dim, DynamicFeedForward(dim, mlp_dim, heads, dropout = dropout, width_mult = width_mult), widths=widths)
            ]))

"""### Vision Transformer"""

class ViT(nn.Module):
    def __init__(
        self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, 
        pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.
        ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.depth = depth

    def forward(self, img, head_mask = None, return_states = False, width_n=None):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        if head_mask is None:
            head_mask = [None] * self.depth
            trans_out = self.transformer(x, head_mask=head_mask, return_states=return_states, width_n=width_n)
        else:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.depth, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=torch.float32)
            trans_out = self.transformer(x, head_mask = head_mask, return_states=return_states, width_n=width_n)
        
        if return_states:
            x, hidden_states = trans_out
        else:
            x = trans_out

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)

        if return_states:
            return x, hidden_states
        else:
            return x

"""### DynaViT"""

class DynaViT(ViT):
    def __init__(
        self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, 
        pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,
        width_mult = 1.0, widths=None
        ):
        super(DynaViT, self).__init__(
            image_size=image_size, patch_size=patch_size, num_classes=num_classes, 
            dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, pool = 'cls', 
            channels = channels, dim_head = dim_head, dropout = dropout, emb_dropout = dropout
        )

        self.transformer = DynamicTransformer(dim, depth, heads, dim_head, mlp_dim, dropout, width_mult = width_mult, widths=widths)

"""### Dyanmic Linear Layer"""

# for rounding to get the heads up to num_heads*width_mult
def round_to_nearest(input_size, width_mult, num_heads, min_value=1):
    new_width_mult = round(num_heads * width_mult)*1.0/num_heads
    input_size = int(new_width_mult * input_size)
    new_input_size = max(min_value, input_size)
    return new_input_size

class DynaLinear(nn.Linear):
    def __init__(self, in_features, out_features, num_heads, bias=True, dyna_dim=[True, True], width_mult=1.0):
        super(DynaLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features
        self.num_heads = num_heads
        self.width_mult = width_mult
        self.dyna_dim = dyna_dim

    def forward(self, input):
        if self.dyna_dim[0]:
            self.in_features = round_to_nearest(self.in_features_max, self.width_mult, self.num_heads)
        if self.dyna_dim[1]:
            self.out_features = round_to_nearest(self.out_features_max, self.width_mult, self.num_heads)
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)

    def set_grad_to_zero(self, width): # to not update parameters in incremental training
        if self.dyna_dim[0]:
            in_features = round_to_nearest(self.in_features_max, width, self.num_heads)
        if self.dyna_dim[1]:
            out_features = round_to_nearest(self.out_features_max, width, self.num_heads)
        self.weight.grad[:self.out_features, :self.in_features] = 0
        if self.bias is not None:
            self.bias.grad[:self.out_features] = 0

def TestDynamicTransformer():
    dim = 64
    depth = 4
    heads = 4
    dim_head = 16
    mlp_dim = 256
    dropout = 0.1
    seq_len = 12
    batch_size = 2
    width_mult = 0.8
    model = DynamicTransformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout, width_mult=width_mult)
    x = torch.randn([batch_size, seq_len, dim])
    out = model(x, head_mask = [None]*depth)
    assert x.shape == out.shape

TestDynamicTransformer()

def TestDynaViT():
    dim = 64
    depth = 4
    heads = 4
    dim_head = 16
    mlp_dim = 256
    dropout = 0.1
    channels = 3
    batch_size = 2
    image_size = 256
    patch_size = 64
    num_classes = 3
    width_mult = 0.8
    model = DynaViT(
        image_size=image_size, patch_size=patch_size, num_classes=num_classes, 
        dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, pool = 'cls', 
        channels = channels, dim_head = dim_head, dropout = dropout, emb_dropout = dropout,
        width_mult = width_mult, widths=5
        )
    x = torch.randn([batch_size, channels, image_size, image_size])
    out = model(x, head_mask = torch.ones([depth, heads]), width_n=4)
    print(model)
    assert x.shape[0] == out.shape[0] and out.shape[1] == num_classes

TestDynaViT()

def all_unit_tests():
    TestDynaViT()
    TestDynamicTransformer()

all_unit_tests()

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
    train_data, eval_data, mode = "full", method = None, width_list = None,
    weights_file = None, **args
    ):
    assert mode in ["full", "width", "height"], "Wrong mode input"

    if "model" not in args:
        print("No pretrained model found, initializing model")
        if method=="difflayernorm":
            model = DynaViT(
                    image_size=args["image_size"], patch_size=args["patch_size"], 
                    num_classes=args["num_classes"], dim=args["dim"], 
                    depth=args["depth"], heads=args["heads"], mlp_dim=args["mlp_dim"], 
                    pool = args["pool"], channels = args["channels"], dim_head = args["dim_head"], 
                    dropout = args["dropout"], emb_dropout = args["emb_dropout"],
                    width_mult = 1.0, widths=len(width_list)
            )
        else:
            model = DynaViT(
                        image_size=args["image_size"], patch_size=args["patch_size"], 
                        num_classes=args["num_classes"], dim=args["dim"], 
                        depth=args["depth"], heads=args["heads"], mlp_dim=args["mlp_dim"], 
                        pool = args["pool"], channels = args["channels"], dim_head = args["dim_head"], 
                        dropout = args["dropout"], emb_dropout = args["emb_dropout"],
                        width_mult = 1.0
            )
    else:
        model = args["model"]
        args["model_path"] = os.path.join(args["model_path"], "retrained")

    model.to(device)

    if weights_file is not None:
        model.load_state_dict(torch.load(weights_file))
    
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    if mode == "full":
        train_model(model, path = os.path.join(args["model_path"], "model.pt"),
                    train_data = train_data, eval_data = eval_data,
                    epochs = args["epochs"], loss_fn = args["loss_fn"],
                    optimizer=optimizer, scheduler=scheduler
                    )
    
    if mode == "width":

        if method == "separate":
            print("Training separately")
            width_list = sorted(width_list)
            for i, width in enumerate(tqdm(width_list, desc="Width", leave=False)):
                print(f"\nWidth: {width}")
                model.apply(lambda m: setattr(m, 'width_mult', width))
                path = os.path.join(args["model_path"], f"model_width_separate_{width}.pt")
                train_model(
                    model, path = os.path.join(args["model_path"], "model.pt"),
                    train_data = train_data, eval_data = eval_data,
                    epochs = args["epochs"], loss_fn = args["loss_fn"],
                    optimizer=optimizer, scheduler=scheduler
                    )
                model = DynaViT(
                    image_size=args["image_size"], patch_size=args["patch_size"], 
                    num_classes=args["num_classes"], dim=args["dim"], 
                    depth=args["depth"], heads=args["heads"], mlp_dim=args["mlp_dim"], 
                    pool = args["pool"], channels = args["channels"], dim_head = args["dim_head"], 
                    dropout = args["dropout"], emb_dropout = args["emb_dropout"],
                    width_mult = 1.0
                )
        
        if method == "naive":
            print("Training Naive")
            width_list = sorted(width_list)
            path = os.path.join(args["model_path"], "model_width_naive.pt")
            train_naive(
                    model, path = path,
                    train_data = train_data, eval_data = eval_data,
                    epochs = args["epochs"], loss_fn = args["loss_fn"],
                    width_list = width_list, optimizer=optimizer, scheduler=scheduler
                    )

        if method == "sandwich":
            print("Training Sandwich")
            width_list = sorted(width_list)
            path = os.path.join(args["model_path"], "model_width_sandwich.pt")
            train_sandwich(
                    model, path = path,
                    train_data = train_data, eval_data = eval_data,
                    epochs = args["epochs"], loss_fn = args["loss_fn"],
                    width_min = width_list[0], width_max = width_list[-1], n_widths=len(width_list), 
                    optimizer=optimizer, scheduler=scheduler
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

def train_sandwich(model, train_data, eval_data, path, epochs, loss_fn,  
                        optimizer, scheduler, width_min = 0.25, width_max = 1, n_widths=5, 
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
