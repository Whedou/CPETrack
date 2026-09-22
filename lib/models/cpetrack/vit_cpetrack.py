""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in:
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929
`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270
The official jax code is released and available at https://github.com/google-research/vision_transformer
DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2021 Ross Wightman

Modified by Botao Ye
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.cpetrack.base_backbone import BaseBackbone
from lib.models.cpetrack.utils import recover_tokens

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            feat, attn = self.attn(self.norm1(x), True)
            x = x + self.drop_path(feat)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class ExpertTokenAlign(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class PixelExpertBank(nn.Module):
    """
    Build RGB-T pixel experts and align them to ViT tokens.

    Experts:
        E0: none, zero image/token
        E1: common = 0.5 * (RGB + TIR)
        E2: diff = phi_c(|RGB - TIR|)
        E3: rel = gate * RGB + (1 - gate) * TIR
        E4: adaptive = phi_a([RGB, TIR, |RGB - TIR|])
    """

    def __init__(self, embed_dim, patch_size, norm_layer, hidden_channels=16):
        super().__init__()
        self.num_experts = 5

        self.diff_pixel = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, kernel_size=1, bias=True),
        )
        self.rel_gate = nn.Sequential(
            nn.Conv2d(9, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.adaptive_pixel = nn.Sequential(
            nn.Conv2d(9, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, kernel_size=1, bias=True),
        )

        self.common_align = ExpertTokenAlign(3, embed_dim, patch_size, norm_layer)
        self.diff_align = ExpertTokenAlign(3, embed_dim, patch_size, norm_layer)
        self.rel_align = ExpertTokenAlign(3, embed_dim, patch_size, norm_layer)
        self.adaptive_align = ExpertTokenAlign(3, embed_dim, patch_size, norm_layer)

    def build_image_experts(self, x):
        rgb = x[:, :3]
        tir = x[:, 3:]

        common = 0.5 * (rgb + tir)
        abs_diff = torch.abs(rgb - tir)
        joint = torch.cat([rgb, tir, abs_diff], dim=1)
        diff = self.diff_pixel(abs_diff)
        gate = self.rel_gate(joint)
        rel = gate * rgb + (1.0 - gate) * tir
        adaptive = self.adaptive_pixel(joint)
        none = torch.zeros_like(common)

        return torch.stack([none, common, diff, rel, adaptive], dim=1)

    def align_token_experts(self, image_experts):
        common_token = self.common_align(image_experts[:, 1])
        diff_token = self.diff_align(image_experts[:, 2])
        rel_token = self.rel_align(image_experts[:, 3])
        adaptive_token = self.adaptive_align(image_experts[:, 4])
        none_token = torch.zeros_like(common_token)

        return torch.stack(
            [none_token, common_token, diff_token, rel_token, adaptive_token], dim=1
        )

    def forward(self, x):
        image_experts = self.build_image_experts(x)
        token_experts = self.align_token_experts(image_experts)
        return token_experts, image_experts


class InputPixelFusion(nn.Module):
    """
    Lightweight input-level expert fusion before the single-stream patch embed.

    The none expert is kept for DPEI, but excluded here to avoid diluting the
    fused image with a zero branch before patch embedding.
    """

    def __init__(self, num_experts, hidden_channels=16):
        super().__init__()
        self.num_experts = num_experts
        self.num_fusion_experts = num_experts - 1
        self.router = nn.Sequential(
            nn.Conv2d(self.num_fusion_experts * 3, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, self.num_fusion_experts),
        )
        self.default_logits = nn.Parameter(torch.zeros(self.num_fusion_experts))
        with torch.no_grad():
            self.default_logits[0] = 2.0

    def forward(self, image_experts):
        fusion_experts = image_experts[:, 1:]
        B, K, C, H, W = fusion_experts.shape
        router_input = fusion_experts.reshape(B, K * C, H, W)
        logits = self.router(router_input) + self.default_logits
        fusion_weights = torch.softmax(logits, dim=1)
        fused = torch.einsum("bk,bkchw->bchw", fusion_weights, fusion_experts)

        weights = image_experts.new_zeros(image_experts.shape[0], self.num_experts)
        weights[:, 1:] = fusion_weights
        return fused, weights


class TemplateTextGuidedTokenRouter(nn.Module):
    """
    Router for Template- and Text-guided Token-wise Expert Injection (TTEI).
    """

    def __init__(self, embed_dim, num_experts, mlp_ratio=0.25, text_feature_dim=0):
        super().__init__()
        self.text_feature_dim = int(text_feature_dim or 0)
        hidden_dim = max(num_experts * 8, int(embed_dim * mlp_ratio))
        input_dim = embed_dim * 3
        if self.text_feature_dim > 0:
            self.text_proj = nn.Sequential(
                nn.Linear(self.text_feature_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )
            input_dim = embed_dim * 5

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, search_tokens, template_tokens, text_feature=None):
        template_cue = template_tokens.mean(dim=1, keepdim=True)
        template_cue = template_cue.expand_as(search_tokens)

        if self.text_feature_dim <= 0:
            router_input = torch.cat(
                [search_tokens, template_cue, search_tokens * template_cue], dim=-1
            )
            return torch.softmax(self.mlp(router_input), dim=-1)

        if text_feature is None:
            text_cue = search_tokens.new_zeros(search_tokens.shape[0], 1, search_tokens.shape[-1])
        else:
            if text_feature.dim() == 1:
                text_feature = text_feature.unsqueeze(0)
            if text_feature.shape[0] == 1 and search_tokens.shape[0] > 1:
                text_feature = text_feature.expand(search_tokens.shape[0], -1)
            if text_feature.shape[0] != search_tokens.shape[0]:
                raise ValueError(
                    "text_feature batch size {} does not match token batch size {}".format(
                        text_feature.shape[0], search_tokens.shape[0]
                    )
                )
            text_feature = text_feature.to(device=search_tokens.device, dtype=search_tokens.dtype)
            text_cue = self.text_proj(text_feature).unsqueeze(1)

        text_cue = text_cue.expand_as(search_tokens)
        router_input = torch.cat(
            [
                search_tokens,
                template_cue,
                text_cue,
                search_tokens * template_cue,
                search_tokens * text_cue,
            ],
            dim=-1,
        )
        return torch.softmax(self.mlp(router_input), dim=-1)


class VisionTransformer(BaseBackbone):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 cross_loc=None, drop_path=None, text_feature_dim=0):
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
        self.depth = depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        self.pixel_expert_bank = PixelExpertBank(embed_dim, patch_size, norm_layer)
        self.input_pixel_fusion = InputPixelFusion(self.pixel_expert_bank.num_experts)
        self.token_routers = nn.ModuleList([
            TemplateTextGuidedTokenRouter(
                embed_dim,
                self.pixel_expert_bank.num_experts,
                text_feature_dim=text_feature_dim,
            )
            for _ in range(depth)
        ])
        self.layer_injection_logits = nn.Parameter(torch.full((depth,), -3.0))
        self.expert_injection_start = 4
        self.expert_injection_end = min(7, depth)

        self.norm = norm_layer(embed_dim)
        self.init_weights(weight_init)

    def _apply_ttei(self, tokens, expert_tokens, layer_idx, lens_z, text_feature=None):
        template_tokens = tokens[:, :lens_z, :]
        search_tokens = tokens[:, lens_z:, :]
        router_weights = self.token_routers[layer_idx](search_tokens, template_tokens, text_feature)
        expert_prior = torch.einsum("bnk,bknc->bnc", router_weights, expert_tokens)
        injection_gate = torch.sigmoid(self.layer_injection_logits[layer_idx])

        search_tokens = search_tokens + injection_gate * expert_prior
        tokens = torch.cat([tokens[:, :lens_z, :], search_tokens], dim=1)
        return tokens, router_weights


    def forward_features(self, z, x, track_query_before=None, keep_rate=None, text_feature=None, **kwargs):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        number = len(z)

        search_expert_tokens, search_image_experts = self.pixel_expert_bank(x)
        x_fused, input_weights_x = self.input_pixel_fusion(search_image_experts)

        x_list = []
        input_weights_z = []

        for i in range(number):
            template_image_experts = self.pixel_expert_bank.build_image_experts(z[i])
            z_fused, input_weights = self.input_pixel_fusion(template_image_experts)
            input_weights_z.append(input_weights)

            z_tokens = self.patch_embed(z_fused)
            z_tokens += self.pos_embed_z
            x_list.append(z_tokens)

        x_tokens = self.patch_embed(x_fused)
        x_tokens += self.pos_embed_x

        x_list.append(x_tokens)

        x_tokens = torch.cat(x_list, dim=1)

        x_tokens = self.pos_drop(x_tokens)
        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        lens_z = lens_z * number
        token_router_weights = []
        layer_search_tokens = []

        for i, blk in enumerate(self.blocks):
            x_tokens = blk(x_tokens)
            if self.expert_injection_start <= i < self.expert_injection_end:
                x_tokens, router_weights = self._apply_ttei(
                    x_tokens,
                    search_expert_tokens,
                    i,
                    lens_z,
                    text_feature=text_feature,
                )
            else:
                search_tokens = x_tokens[:, lens_z:, :]
                none_weights = search_tokens.new_ones(search_tokens.shape[0], search_tokens.shape[1], 1)
                other_weights = search_tokens.new_zeros(
                    search_tokens.shape[0],
                    search_tokens.shape[1],
                    self.pixel_expert_bank.num_experts - 1,
                )
                router_weights = torch.cat([none_weights, other_weights], dim=-1)
            layer_search_tokens.append(x_tokens[:, lens_z:, :].detach().to(dtype=torch.float16))
            token_router_weights.append(router_weights)

        x_tokens = recover_tokens(x_tokens, lens_z, lens_x, mode=self.cat_mode)
        len_zx = [lens_z, lens_x]
        aux_dict = {
            "attn": None,
            "input_router_weights_x": input_weights_x.detach(),
            "input_router_weights_z": torch.stack(input_weights_z, dim=1).detach(),
            "token_router_weights": torch.stack(token_router_weights, dim=1).detach(),
            "layer_injection_gate": torch.sigmoid(self.layer_injection_logits.detach()),
            "layer_search_tokens": torch.stack(layer_search_tokens, dim=1),
        }
        return self.norm(x_tokens), aux_dict, len_zx

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_xmpl=True), self)
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


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_xmpl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_xmpl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_xmpl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_xmpl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_xmpl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    print('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = VisionTransformer(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_patch16_224(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) with PointFlow between RGB and T search regions.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=False, **model_kwargs)
    return model


def vit_small_patch16_224(pretrained=False, **kwargs):
    """
    ViT-Small model (ViT-S/16) with PointFlow between RGB and T search regions.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


def vit_tiny_patch16_224(pretrained=False, **kwargs):
    """
    ViT-Tiny model (ViT-S/16) with PointFlow between RGB and T search regions.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model
