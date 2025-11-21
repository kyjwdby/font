import torch
import torch.nn.functional as F
# import cairosvg
# from data_utils.common_utils import trans2_white_bg
from PIL import Image
import numpy as np

def sequence_mask(lengths, max_len=None):
    batch_size=lengths.numel()
    max_len=max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
    .type_as(lengths)
    .unsqueeze(0).expand(batch_size,max_len)
    .lt(lengths.unsqueeze(1)))
