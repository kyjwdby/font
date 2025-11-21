import math
import torch
import torch.nn as nn

######################## added self
import torch.nn.functional as F
def image_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, 1e-9)
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, value)


class StyleAttention(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 1024
        self.bone_k = nn.Linear(hidden_size, hidden_size//4)
        self.bone_v = nn.Linear(hidden_size, hidden_size)
        self.image_q1 = nn.Linear(hidden_size, hidden_size//4)
        # self.image_q2 = nn.Linear(output_channels, output_channels//4)
        # self.image_k1 = nn.Linear(output_channels, output_channels)
        # self.image_k2 = nn.Linear(output_channels, output_channels)
        self.attn_LN1 = nn.LayerNorm(hidden_size)
        # self.attn_LN2 = nn.LayerNorm(output_channels)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, style_hidden_states, bone_feature):
        bone_v = self.bone_v(bone_feature)
        bone_k = self.bone_k(bone_feature)
        image_q1 = self.image_q1(style_hidden_states)
        # image_k1 = self.image_k1(style_hidden_states)
        style_hidden_states = self.attn_LN1(image_attention(image_q1, bone_k, bone_v, dropout=self.drop) + style_hidden_states)
        return style_hidden_states



class StyleModulator(nn.Module):
    def __init__(self, style_dim=(9+35)*1024, num_layers=4+1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(style_dim, 128),
            nn.SiLU(),
            nn.Linear(128, num_layers)
        )

    def forward(self, style_feat):
        B, C, _, N = style_feat.shape
        weights = torch.sigmoid(self.mlp(style_feat.view(B, -1)))  # (B, num_layers)
        return weights


