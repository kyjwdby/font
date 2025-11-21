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


class IDSEncoder(nn.Module):
  def __init__(self, max_len: int, n_embd: int, corpus='', input_mode='ids', ids_mode='all'):
    super().__init__()
    assert input_mode in {'ch', 'ids'}
    assert ids_mode in {'all', 'radical', 'stroke'}
    assert isinstance(cn.IDC, tuple), 'cn.IDC must be tuple (orderly)'

    self.input_mode = input_mode
    self.ids_mode = ids_mode
    self.special_tokens = ('pad', 'sep', ) if ids_mode == 'all' else ('pad', )
    self.vocabulary_map = dict()
    self.max_len = max_len
    corpus = corpus or valid_characters.valid_ch
    self.r_map, _, self.r_base = cn.resolve_IDS_babelstone(corpus, 'radical')
    self.s_map, _, self.s_base = cn.resolve_IDS_babelstone(corpus, 'stroke')

    count = 0
    if ids_mode == 'all':
      tmp = itertools.chain(self.r_base.keys(), self.s_base.keys())
    else:
      tmp = getattr(self, f'{ids_mode[0]}_base').keys()
    tmp = sorted(set(tmp))
    tmp = itertools.chain(self.special_tokens, cn.IDC, tmp)
    for token in tmp:
      if token in self.vocabulary_map:
        continue
      self.vocabulary_map[token] = count
      count += 1
    
    self.num_tokens = count
    self.embedding = nn.Embedding(count, n_embd)
    self.embedding2 = nn.Embedding(count, n_embd)
    self.ids_map = dict()
    for i, token in enumerate(cn.IDS_VOCABULARY[len(cn.IDC):]):
      self.ids_map[token] = i
    # self.ids_mutator = IDSMutator(n_variation, cache_size=n_variation - 1)
    # encode_layer = nn.TransformerEncoderLayer(n_embd, nhead=8, dim_feedforward=n_embd * 2, batch_first=True)
    # self.transformer = nn.ModuleDict(dict(
    #     h=nn.TransformerEncoder(encode_layer, num_layers=4),
    #     pe=nn.Embedding(max_len, n_embd),
    # ))

  def set_device(self, device: torch.device):
    self.device = device
    self.to(device)

  def query_ids(self, ch:str, ids_mode=None) -> tuple[tuple[str]]:
    ids_mode = ids_mode or self.ids_mode

    s, r = None, None
    if ids_mode != 'radical':
      s = self.s_map[ch]
      s = self.triplet_equal_ids(s)
    if ids_mode != 'stroke':
      r = self.r_map[ch]
      r = self.triplet_equal_ids(r)
    
    if s is None:
      ids = r
    elif r is None:
      ids = s
    else:
      ids = []
      for i1 in r:
        i1 += ('sep', )
        for i2 in s:
          t = i1
          t += i2
          ids.append(t)
      ids = tuple(ids)
      
    return ids
  
  def triplet_equal_ids(self, ids: tuple) -> tuple[tuple]:
    level, _ = self.level_analysis(ids, depth=2)
    idc = None
    match level:
      case '⿲', a, b, c:
        idc = 0
      case '⿰', ('⿰', *_), ('⿰', *_):
        ...
      case '⿰', ('⿰', a, b), c:
        idc = 0
      case '⿰', a, ('⿰', b, c):
        idc = 0
      case '⿳', a, b, c:
        idc = 1
      case '⿱', ('⿱', *_), ('⿱', *_):
        ...
      case '⿱', ('⿱', a, b), c:
        idc = 1
      case '⿱', a, ('⿱', b, c):
        idc = 1

    if idc is None:
      return (ids, )
    t2, t3 = (('⿰', '⿲'), ('⿱', '⿳'))[idc]

    ret = lambda x: utils.chain_sequence(*x) if isinstance(x, tuple) else x
    a, b, c = tuple(map(ret, (a, b, c)))
    ret = (
      tuple(utils.chain_sequence(t3, a, b, c)),
      tuple(utils.chain_sequence(t2, a, t2, b, c)),
      tuple(utils.chain_sequence(t2, t2, a, b, c)),
    )
    return ret

  @staticmethod
  def level_analysis(ids, depth=-1, _d=1) -> tuple:

    l_ids = len(ids)
    if l_ids <= 1:
      return (ids, l_ids, )

    main_idc = ids[0]
    assert main_idc in cn.IDC, f'illegal ids {ids}'
    level = [main_idc]
    i = 1

    while i < l_ids:
      if len(level) == cn.N_IDC_COMPS[main_idc]+1:
        break
      if ids[i] in cn.IDC:
        new_level, new_i = IDSEncoder.level_analysis(ids[i:], depth, _d+1)
        if depth > 0 and _d >= depth:
          new_level = tuple(utils.chain_sequence(*new_level))
        level.append(new_level)
        i += new_i
      else:
        level.append(ids[i])
        i += 1
  
    assert len(level) == cn.N_IDC_COMPS[main_idc]+1, f'illegal {ids=}, {level=}'
    return tuple(level), i

  def count_idc(self, ids_list: list[tuple]) -> torch.Tensor:
    res = []
    for ids in ids_list:
      for c in filter(lambda x: x not in cn.IDC, ids):
        ic[self.ids_map[c]] = 1
        ic[self.ids_map[c]] += 1
      ic = torch.as_tensor(ic, device=self.device)
      res.append(ic)
    res = torch.stack(res, dim=0)
    return res

  def coverage(self, targets:tuple, sources:tuple) -> torch.Tensor:
    def count(target, source):
      ti = match_cnt = 0
      while True:
        if ti == ti_max:
          break
        tc = target[ti]
        if tc not in cn.IDC:
          ti += 1
          continue

        si, si_max = 0, len(source)
        while True:
          if si >= si_max:
            ti += 1
            break
          if source[si] != tc:
            si += 1
            continue
      
          ti2, si2 = ti, si
          while ti2<ti_max and si2<si_max and target[ti2] == source[si2]:
            ti2 += 1
            si2 += 1

          if ti2==ti_max or target[ti2] in cn.IDC:
            match_cnt += ti2-ti
            ti = ti2
            break
          si += 1
      return match_cnt

    def get_ids(x):
      if self.input_mode == 'ch':
        ids = self.query_ids(x, 'stroke')[0]
      else:
        ids = x
      return ids
    
    res = []
    for t, source in zip(targets, sources):
      res_row = []
      t_ids = get_ids(t)
      ti_max = len(t_ids)
      for s in source:
        s_ids = get_ids(s)
        cnt = count(t_ids, s_ids)
        res_row.append(cnt / ti_max)
      res.append(res_row)
    res = torch.as_tensor(res, device=self.device)  # (bs, n_ref)
    return res

  def embed(self, x: str) -> torch.Tensor:
    if self.input_mode == 'ch':
      ids = self.query_ids(x)
      rid = random.randint(0, len(ids)-1)
      ids = ids[rid]
      rid = f'{x}-{rid}'
    else:
      ids, rid = x, None

    tokens = [self.vocabulary_map['pad']] * self.max_len
    count = 0
    for s in ids:
      tokens[count] = self.vocabulary_map[s]
      count += 1
    tokens = torch.as_tensor(tokens, dtype=torch.long, device=self.device)

    # mask = torch.zeros(self.max_len, device=self.device)
    # mask[count:] = -torch.inf
    return self.embedding(tokens), self.embedding2(tokens)

  def forward(self, sequence: tuple) -> tuple[torch.Tensor]:
    embed = []
    embed2 = []
    for data in sequence:
      e, e2 = self.embed(data)
      embed.append(e)
      embed2.append(e2)
    embed = torch.stack(embed, dim=0)  # (bs*num, max_len, n_embed)
    embed2 = torch.stack(embed2, dim=0)
    return embed, embed2
