from functools import partial
import random
import copy
import math
import itertools

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from torchvision.models import mobilenetv3
from torchvision.models import efficientnet
from torchvision.ops.misc import Conv2dNormActivation

from .data.adapter import VQAdapter
from .data import cn, valid_characters
from .modules.blocks import ConvBlock, ResBlock, DropKeyMask
from .util import utils



class CompontStyleFusion(nn.Module):
  def __init__(self, adapter: VQAdapter, c_out=None, l_ids=45, dropout_p=0):
    super().__init__()
    ### from QuantExtEncoder
    self.adapter = adapter
    self.c_in = self.adapter.vqgan.quantize.e_dim
    self.c_out = c_out
    #####################
    self.n_head = 8
    self.q_linear = nn.Linear(c_out, c_out, bias=False)
    self.kv_linear = nn.Linear(c_out, 2 * c_out, bias=False)
    self.c_proj = nn.Linear(c_out, c_out, bias=False)
    self.layer_norm = nn.LayerNorm(c_out, bias=False, eps=1e-6)
    self.wpe = nn.Embedding(l_ids, c_out)
    self.stem = self._init_enc(dropout_p=dropout_p)
    self.cl_head = nn.Sequential(
      nn.Linear(c_out, 1),  # [b, 2, 256, 1]
      nn.Flatten(-2, -1),  # [b, 2, 256]
      nn.LayerNorm(256),
      # nn.LayerNorm(1024),
      nn.SiLU(True),
      nn.Dropout(dropout_p, inplace=True),
    )
    self.cl_fc = nn.Linear(256, c_out)  # [b, 2, c_out]
    self.dropout_p = dropout_p if self.training else 0
    self.dropkey_mask = DropKeyMask(dropout_p, is_causal=False)
    self.ln_q = nn.LayerNorm(c_out // self.n_head)
    self.ln_k = nn.LayerNorm(c_out // self.n_head)

    for m in self.modules():
      if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
          torch.nn.init.zeros_(m.bias)
      elif isinstance(m, nn.Embedding):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    ### from MoCoWrapper
    self._build_projector_and_predictor_mlps(c_out, 1024)
    ##################### self added
    self.style_improver = torch.nn.Linear(c_out, 1024)  # 256->1024
    self.style_query = nn.Parameter(torch.randn(9, 1024))
    self.style_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
  
  ### from QuantExtEncoder
  def _init_enc(self, dropout_p=0):
    ConvBlk = partial(ConvBlock, norm='in', pad_type='reflect', dropout=dropout_p)
    ResBlk = partial(ResBlock, norm='in', dropout=dropout_p)
    C = 32
    return nn.Sequential(
      ConvBlk(self.c_in, C),  # 16x16x32
      ConvBlk(C * 1, C * 2),  # , downsample=True
      ConvBlk(C * 2, C * 4),  # , downsample=True
      ResBlk(C * 4, C * 4),
      ResBlk(C * 4, C * 4),
      ResBlk(C * 4, C * 8),  # , downsample=True
      ResBlk(C * 8, self.c_out)  # 16x16x256
    )
  
  def set_device(self, device: torch.device):
    self.device = device
    self.adapter.set_device(device)
    self.to(device)

  def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
    x = x.view(-1, x.shape[-1])  # torch.Size([48, 256])
    x = self.adapter.lookup_quant(x)  # torch.Size([48, 4, 16, 16])
    return x
  #######################
  ### from MoCoWrapper
  def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    mlp = []
    for l in range(num_layers):
      dim1 = input_dim if l == 0 else mlp_dim
      dim2 = output_dim if l == num_layers - 1 else mlp_dim
      mlp.append(nn.Linear(dim1, dim2, bias=False))
      if l < num_layers - 1:
        mlp.append(nn.BatchNorm1d(dim2))
        mlp.append(nn.ReLU(inplace=True))
      elif last_bn:
        mlp.append(nn.BatchNorm1d(dim2, affine=False))
    # mlp.append(Rearrange('(b p) c -> b p c', p=2))
    return nn.Sequential(*mlp)
  
  def _build_projector_and_predictor_mlps(self, dim=256, mlp_dim=4096):
    hidden_dim = self.cl_fc.weight.shape[1]
    del self.cl_fc
    self.cl_fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
  #######################
   
  def _structure_style_aggregation(self, ids:torch.Tensor, feat:torch.Tensor) -> torch.Tensor:
    ids += self.wpe(torch.arange(0, ids.shape[1], dtype=torch.long, device=ids.device))
    q = self.q_linear(ids)  # [b*2, len, c]
    q = rearrange(q, 'b l (h c) -> b h l c', h=self.n_head)
    k, v = self.kv_linear(feat).split(self.c_out, dim=2)  # [b*2, 256*n//2, 256]
    k = rearrange(k, 'b hw (h c) -> b h hw c', h=self.n_head)
    v = rearrange(v, 'b hw (h c) -> b h hw c', h=self.n_head)
    q, k = self.ln_q(q), self.ln_k(k)

    mask = self.dropkey_mask(1, 1, q.shape[2], k.shape[2], device=ids.device)
    v = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    v = rearrange(v, 'b h l c -> b l (h c)')  # [b*2, len, c]
    return v

  def forward(self, indices: torch.Tensor, ids: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
    # b: batchsize; n: num_refs
    x: torch.Tensor = self.encode_indices(indices)  # [b*n, 4, 16, 16]
    n_ref = indices.shape[1]  # n

    x = self.stem(x)  # [b*n, 256, 16, 16]
    x = rearrange(x, '(b n) c h w -> b n (h w) c', n=n_ref)  # [b, n, 256, c]
    sim *= 3
    # 根据相似度计算加权平均风格
    x_g = torch.einsum('b n f c, b n -> b f c', x, sim.softmax(dim=1))  # [b, 256, c]

    x = rearrange(x, 'b n f c -> b (n f) c')  # [b, 256*n, c]
    x = self._structure_style_aggregation(ids, x)  # [b, len, c]
    x_l = self.c_proj(x)  # [b, len, c]
    x = torch.concat((x_l, x_g), dim=1)  # [b, len+256, c]
    x_sss = self.layer_norm(x)  # [b, len+256, c]
    ##################### self added
    # x_fusion, _ = self.style_attn(query=x_l, key=x_g, value=x_g)  # [B, len, 256]
    x_fusion = x_l
    x_fusion_improve = self.style_improver(x_fusion)  # [B, len, 1024]

    if not self.training:
      return x_sss, None

    x_cl = self.cl_fc(self.cl_head(x_g))  # [b, c]
    # return x_sss, x_cl
    return x_fusion_improve, x_cl