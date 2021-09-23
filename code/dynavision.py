"""### Importing Libraries"""

import torch
from torch import nn, einsum
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from torch.optim import Adam, lr_scheduler
from torchvision import transforms

import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pickle
import os
import re

from timm.models.layers import DropPath, PatchEmbed, trunc_normal_, lecun_normal_

from functools import partial

from tqdm import tqdm

# helpers
def show_torch_img(image):
    plt.imshow(transforms.ToPILImage()(image))

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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., attn_dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head

        self.attend = nn.Softmax(dim = -1)
        self.attn_dropout = nn.Dropout(attn_dropout)
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
        attn = self.attn_dropout(attn)

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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., attn_dropout=0., width_mult = 1.0):
        super(DynamicAttention, self).__init__(dim, heads, dim_head, dropout=dropout, attn_dropout=attn_dropout)
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
        
        attn = self.attn_dropout(attn)

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
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_drop=0., drop_paths=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                nn.Identity() if drop_paths is None else DropPath(drop_paths[i]),
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, attn_dropout=attn_drop)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, head_mask = None, return_states = False, width_n=None):
        hidden_states = []
        for i, (drop_path, attn, ff) in enumerate(self.layers):
            x = drop_path(attn(x, head_mask = head_mask[i], width_n=width_n)) + x
            x = drop_path(ff(x, width_n=width_n)) + x
            hidden_states.append(x)
        if return_states:
            return x, hidden_states
        return x

class DynamicTransformer(Transformer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., width_mult = 1.0, widths=None, attn_drop=0., drop_paths=None):
        super(DynamicTransformer, self).__init__(dim, depth, heads, dim_head, mlp_dim, dropout=dropout, attn_drop=attn_drop)
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                nn.Identity() if drop_paths is None else DropPath(drop_paths[i]),
                PreNorm(dim, DynamicAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, width_mult = width_mult, attn_dropout=attn_drop), widths=widths),
                PreNorm(dim, DynamicFeedForward(dim, mlp_dim, heads, dropout = dropout, width_mult = width_mult), widths=widths)
            ]))

"""### Vision Transformer"""

class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        dim_head = embed_dim // num_heads
        mlp_dim = embed_dim * mlp_ratio
        self.blocks = Transformer(embed_dim, depth, num_heads, dim_head, mlp_dim, drop_rate, attn_drop=attn_drop_rate, drop_paths=dpr)
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x

class DynamicVisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', width_mult = 1.0, widths=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        dim_head = embed_dim // num_heads
        mlp_dim = embed_dim * mlp_ratio
        self.blocks = DynamicTransformer(embed_dim, depth, num_heads, dim_head, mlp_dim, drop_rate, attn_drop=attn_drop_rate, drop_paths=dpr, width_mult = width_mult, widths=widths)
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()


    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x

