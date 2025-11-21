from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils.resnet import resnet34
import math
import torch.distributions as DIS
import numpy as np


class encoder_block(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, img_size):
        super(encoder_block, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        self.LN1 = nn.LayerNorm((output_channels, img_size, img_size))
        self.ReLU = nn.LeakyReLU(0.1, inplace=True)

        self.conv2 = nn.Conv2d(input_channels, output_channels, 1, stride, 0)
        self.LN2 = nn.LayerNorm((output_channels, img_size, img_size))

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.LN1(x1)
        x1 = self.ReLU(x1)

        x2 = self.conv2(x)
        x2 = self.LN2(x2)
        return x1 + x2


class encoder(nn.Module):
    def __init__(self, input_channels, img_size, ngf):
        super(encoder, self).__init__()
        channels = [input_channels, ngf, ngf, 2*ngf, 2*ngf, 4*ngf, 4*ngf, 8*ngf, 8*ngf]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3]
        strides = [1, 2, 2, 2, 2, 2, 2, 2]
        self.blocks = nn.ModuleList()
        for i in range(len(kernel_sizes)):
            img_size = (img_size + kernel_sizes[i] // 2 * 2 - (kernel_sizes[i]-1) - 1) // strides[i] + 1
            block = encoder_block(channels[i], channels[i+1], kernel_sizes[i], strides[i], img_size)
            self.blocks.append(block)

    def forward(self, x):
        output = []
        for block in self.blocks:
            x = block(x)
            output.append(x)
        return x, output


def image_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, 1e-9)
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, value)


class decoder_block(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, output_padding, img_size, attn):
        super(decoder_block, self).__init__()
        padding = kernel_size // 2
        self.deconv1 = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, output_padding)
        self.LN1 = nn.LayerNorm((output_channels, img_size, img_size))
        self.ReLU = nn.LeakyReLU(0.1, inplace=True)

        self.deconv2 = nn.ConvTranspose2d(input_channels, output_channels, 1, stride, 0, output_padding)
        self.LN2 = nn.LayerNorm((output_channels, img_size, img_size))
        self.attn = attn
        if attn:
            self.seq_k = nn.Linear(512, output_channels//4)
            self.seq_v = nn.Linear(512, output_channels)
            self.image_q1 = nn.Linear(output_channels, output_channels//4)
            self.image_q2 = nn.Linear(output_channels, output_channels//4)
            self.image_k1 = nn.Linear(output_channels, output_channels)
            self.image_k2 = nn.Linear(output_channels, output_channels)
            self.attn_LN1 = nn.LayerNorm(output_channels)
            self.attn_LN2 = nn.LayerNorm(output_channels)
            self.drop = nn.Dropout(p=0.1)

    def forward(self, x, seq_feat):
        x1 = self.deconv1(x)
        x1 = self.LN1(x1)
        x1 = self.ReLU(x1)

        x2 = self.deconv2(x)
        x2 = self.LN2(x2)
        if self.attn:
            seq_v = self.seq_v(seq_feat)
            seq_k = self.seq_k(seq_feat)

            B, C, H, W = x1.shape 
            x1 = x1.view(B, C, H*W).transpose(-2, -1)
            image_q1 = self.image_q1(x1)
            image_k1 = self.image_k1(x1)
            attn_x1 = self.attn_LN1(image_attention(image_q1, seq_k, seq_v, dropout=self.drop) + x1)
            x1 = image_attention(attn_x1, image_k1, x1, dropout=self.drop)
            x1 = x1.transpose(-2, -1).view(B, C, H, W)

            B, C, H, W = x2.shape 
            x2 = x2.view(B, C, H*W).transpose(-2, -1)
            image_q2 = self.image_q2(x2)
            image_k2 = self.image_k2(x2)
            attn_x2 = self.attn_LN2(image_attention(image_q2, seq_k, seq_v, dropout=self.drop) + x2)
            x2 = image_attention(attn_x2, image_k2, x2, dropout=self.drop)
            x2 = x2.transpose(-2, -1).view(B, C, H, W)
            
            x1 = self.LN1(x1)
            x1 = self.ReLU(x1)
            x2 = self.LN2(x2)
            x2 = self.ReLU(x2)
        return x1 + x2


class decoder(nn.Module):
    def __init__(self, output_channels, img_size, ndf):
        super(decoder, self).__init__()
        channels = [8*ndf, 8*ndf, 4*ndf, 4*ndf, 2*ndf, 2*ndf, ndf, ndf]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 3]
        strides = [2, 2, 2, 2, 2, 2, 2]
        output_paddings = [1, 1, 1, 1, 1, 1, 1]
        attn = [0, 0, 0, 1, 1, 1, 0]
        self.blocks = nn.ModuleList()
        for i in range(len(kernel_sizes)):
            img_size = (img_size - 1) * strides[i] - kernel_sizes[i] // 2 * 2 + kernel_sizes[i] + output_paddings[i]
            block = decoder_block(2*channels[i], channels[i+1], kernel_sizes[i], strides[i], output_paddings[i], img_size, attn[i])
            self.blocks.append(block)
        self.torgb2 = nn.Conv2d(ndf, 1, 3, 1, 1)

    def forward(self, x, cf, seq_feat):
        cnt = 1
        for block in self.blocks:
            x = torch.cat((x, cf[-cnt]), dim=1)
            cnt += 1
            x = block(x, seq_feat)
        x = self.torgb2(x)
        return torch.sigmoid(x)


class image_branch(nn.Module):
    def __init__(self, config):
        super(image_branch, self).__init__()
        if 'dml' in config:
            self.dml = config['dml']
        else:
            self.dml = False
        if 'skip' in config:
            self.skip = config['skip']
        else:
            self.skip = False
        self.content_encoder = encoder(3, 128, 64)
        
        self.style_encoder = resnet34(pretrained=True, pool=True)
        
        self.style_mod = nn.Linear(512, 256, bias=False)
        self.style_demod = nn.Linear(256, 512, bias=False)
        
        self.decoder = decoder(64, 1, 64)
        
        self.temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        if self.dml:
            self.dml_fc_feat_common = nn.Linear(512, 100)
            self.dml_fc_mod_common = nn.Linear(256, 100)
        else:
            self.dml_fc_feat_common = None
            self.dml_fc_mod_common = None

    def encode(self, style):
        B, C, L, L = style.shape
        return self.style_encoder(style).view(B, -1)

    def forward(self, content, style, seq_feat, skip=False):
        content_feat, content_feat_list = self.content_encoder(content)
        
        B, N, C, L, L = style.shape
        style = style.view(B*N, C, L, L)
        style_feat = self.style_encoder(style)
        style_feat = style_feat.view(B, N, -1).mean(1).squeeze()
        
        style_mod = F.normalize(self.style_mod(style_feat), dim=1)
        style_demod = self.style_demod(style_mod)
        
        seq_feat = seq_feat.detach()

        if self.skip:
            fake_image = self.decoder(style_feat[:, :, None, None], content_feat_list, seq_feat)
        else:
            fake_image = self.decoder(style_demod[:, :, None, None], content_feat_list, seq_feat)
        
        temp = self.temp.exp()
        
        if self.dml == True and self.skip==False:
            dml_feat = self.dml_fc_feat_common(F.normalize(style_feat, dim=1))
            dml_mod = self.dml_fc_mod_common(F.normalize(style_mod, dim=1))
        else:
            dml_mod = None
            dml_feat = None
       
        output = {}
        output['fake'] = fake_image
        output['style_feat'] = style_feat
        output['style_mod'] = style_mod
        output['style_demod'] = style_demod
        output['temp'] = temp
        output['dml_mod'] = dml_mod
        output['dml_feat'] = dml_feat
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
    # def __init__(self, d_model, dropout=0.1, max_len=1000):   #####
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class style_seq_encoder(nn.Module):
    def __init__(self, config):
        super(style_seq_encoder, self).__init__()
        self.dml = config.dml
        self.skip = config.skip
        self.hidden_size = config.hidden_size

        self.pe = PositionalEncoding(self.hidden_size)

        style_encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=8, batch_first=True)
        # self.style_map = nn.Linear(3, self.hidden_size)
        # self.style_map = nn.Linear(12, self.hidden_size)   #####
        self.style_map = nn.Linear(3, self.hidden_size)   #####
        self.style_encoder = nn.TransformerEncoder(style_encoder_layer, num_layers=6)
    
    def forward(self, style):
        # 处理style可能是列表或包含None的情况
        if isinstance(style, list):
            # 过滤掉None值，并确保列表不为空
            valid_styles = [s for s in style if s is not None]
            if not valid_styles:
                # 如果没有有效元素，返回默认值
                return torch.zeros(0, 0, self.hidden_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            style = torch.stack(valid_styles)
        
        B, N, L, _ = style.shape
        style = self.style_map(style).view(B*N, L, -1) * math.sqrt(self.hidden_size)
        style_pe = self.pe(style)
        style_feat = self.style_encoder(style_pe)
        style_feat = style_feat.view(B, N*L, -1)
        # style_feat = style_feat.mean(1)

        return style_feat



class seq_branch(nn.Module):
    def __init__(self, config):
        super(seq_branch, self).__init__()
        # if 'dml' in config:
        #     self.dml = config['dml']
        # else:
        #     self.dml = False
        # if 'skip' in config:
        #     self.skip = config['skip']
        # else:
        #     self.skip = False
        self.dml = config.dml
        self.skip = config.skip
        self.hidden_size = config.hidden_size

        self.pe = PositionalEncoding(self.hidden_size)

        content_encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=8, batch_first=True)
        # self.content_map = nn.Linear(3, self.hidden_size)
        # self.content_map = nn.Linear(12, self.hidden_size)   #####
        self.content_map = nn.Linear(3, self.hidden_size)   #####
        self.content_encoder = nn.TransformerEncoder(content_encoder_layer, num_layers=6)

        style_encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=8, batch_first=True)
        # self.style_map = nn.Linear(3, self.hidden_size)
        # self.style_map = nn.Linear(12, self.hidden_size)   #####
        self.style_map = nn.Linear(3, self.hidden_size)   #####
        self.style_encoder = nn.TransformerEncoder(style_encoder_layer, num_layers=6)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=8, batch_first=True)
        # self.decoder_map2 = nn.Linear(6, self.hidden_size)
        # self.decoder_map2 = nn.Linear(12, self.hidden_size)   #####
        self.decoder_map2 = nn.Linear(3, self.hidden_size)   #####
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        self.pred_pi = nn.Linear(self.hidden_size, 20)
        self.pred_mu = nn.Linear(self.hidden_size, 20 * 2)
        self.pred_sigma = nn.Linear(self.hidden_size, 20 * 2)
        # self.pred_mu = nn.Linear(self.hidden_size, 20 * 11)
        # self.pred_sigma = nn.Linear(self.hidden_size, 20 * 11)
        self.pred_rho = nn.Linear(self.hidden_size, 20)
        ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
        # nn.init.xavier_uniform_(self.pred_pi.weight, gain=1.0)
        # nn.init.zeros_(self.pred_pi.bias)   # 保证初始时各分量几乎均匀
        # nn.init.xavier_uniform_(self.pred_mu.weight, gain=0.01)
        # nn.init.zeros_(self.pred_mu.bias)
        # nn.init.xavier_uniform_(self.pred_sigma.weight, gain=0.01)
        # nn.init.constant_(self.pred_sigma.bias, np.log(1.0))  # 初始 σ ≈ 1
        # nn.init.xavier_uniform_(self.pred_rho.weight, gain=0.01)# 权重小一点，避免 tanh 输入太大直接饱和
        # nn.init.zeros_(self.pred_rho.bias)# bias = 0
        ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
        self.pred_label2 = nn.Linear(self.hidden_size, 4)
        
        self.style_mod = nn.Linear(self.hidden_size, 256, bias=False)
        self.style_demod = nn.Linear(256, self.hidden_size, bias=False)
        
        self.temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if self.dml:
            self.dml_fc_feat_common = nn.Linear(self.hidden_size, 100)
            self.dml_fc_mod_common = nn.Linear(256, 100)
        else:
            self.dml_fc_feat_common = None
            self.dml_fc_mod_common = None
       
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, content, style, target, device):
        content = self.content_map(content) * math.sqrt(self.hidden_size)
        content_pe = self.pe(content)
        content_feat = self.content_encoder(content_pe)

        B, N, L, _ = style.shape
        style = self.style_map(style).view(B*N, L, -1) * math.sqrt(self.hidden_size)
        style_pe = self.pe(style)
        style_feat = self.style_encoder(style_pe)
        style_feat = style_feat.view(B, N*L, -1)
        style_feat = style_feat.mean(1)
                
        if self.skip:
            style_mod = None
            style_demod = None
            memory = torch.cat((style_feat[:, None, :], content_feat), 1)
        else:
            style_mod = F.normalize(self.style_mod(style_feat), dim=1)
            style_demod = self.style_demod(style_mod)
            memory = torch.cat((style_demod[:, None, :], content_feat), 1)
        
        B, L2, _ = target[:, :-1].shape
        tgt_mask = self._generate_square_subsequent_mask(L2).to(device)

        target = self.decoder_map2(target) * math.sqrt(self.hidden_size)
        target = self.pe(target)
        seq_feat = self.decoder(target[:, :-1], memory, tgt_mask=tgt_mask)

        pi = self.pred_pi(seq_feat)
        pi = F.softmax(pi, dim=-1)

        mu = self.pred_mu(seq_feat)
        # mu = mu.view(B, L2, 20, 2)
        mu = mu.view(B, L2, 20, -1)

        sigma = self.pred_sigma(seq_feat)
        sigma = torch.minimum(torch.tensor(500.).to(sigma.device), torch.exp(sigma))
        # sigma = sigma.view(B, L2, 20, 2)
        sigma = sigma.view(B, L2, 20, -1)

        rho = self.pred_rho(seq_feat)
        rho = torch.tanh(rho)
        # rho = 0.9 * torch.tanh(rho)  ### !!!!!!!!!!!!!!!!!!!!!!

        label = self.pred_label2(seq_feat)

        temp = self.temp.exp()
        
        output = {}
        if self.dml == True and self.skip==False:
            dml_feat = self.dml_fc_feat_common(F.normalize(style_feat, dim=1))
            dml_mod = self.dml_fc_mod_common(F.normalize(style_mod, dim=1))
        else:
            dml_mod = None
            dml_feat = None
        
        output = {}
        output['fake'] = (pi, mu, sigma, rho, label)
        output['style_feat'] = style_feat
        # output['style_mod'] = style_mod
        # output['style_demod'] = style_demod
        # output['temp'] = temp
        # output['dml_mod'] = dml_mod
        # output['dml_feat'] = dml_feat
        output['seq_feat'] = seq_feat

        return output

    def generate(self, content, style, target, device, skip=False, greedy=True):
        def sample_point(seq, device, greedy=True):
            pi, mu, sigma, rho = seq
            if greedy:
                return torch.gather(mu, 2, pi.argmax(-1)[:, :, None, None].repeat(1, 1, 1, 2))[:, :, 0, :]
            with torch.no_grad():
                B, L, n, _ = mu.shape
                scale_tril = torch.zeros((B, L, n, 2, 2)).to(device)
                s0 = sigma[:, :, :, 0]
                s1 = sigma[:, :, :, 1]
                scale_tril[:, :, :, 0, 0] = s0
                scale_tril[:, :, :, 1, 0] = rho * s1
                scale_tril[:, :, :, 1, 1] = s1 * torch.sqrt((1 - rho ** 2))
                mix = DIS.Categorical(probs=pi)
                comp = DIS.MultivariateNormal(loc=mu, scale_tril=scale_tril)
                gmm = DIS.MixtureSameFamily(mix, comp)
                p = gmm.sample()
                return p

        content = self.content_map(content) * math.sqrt(self.hidden_size)
        content_pe = self.pe(content)
        content_feat = self.content_encoder(content_pe)

        B, N, L, _ = style.shape
        style = self.style_map(style).view(B*N, L, -1) * math.sqrt(self.hidden_size)
        style_pe = self.pe(style)
        style_feat = self.style_encoder(style_pe)
        style_feat = style_feat.view(B, N*L, -1)
        style_feat = style_feat.mean(1)

        if skip:
            style_mod = None
            style_demod = None
            memory = torch.cat((style_feat[:, None, :], content_feat), 1)
        else:
            style_mod = F.normalize(self.style_mod(style_feat), dim=1)
            style_demod = self.style_demod(style_mod)
            memory = torch.cat((style_demod[:, None, :], content_feat), 1)

        B, L2, _ = target.shape

        fake = target[:, :1]
        index = torch.arange(0, B).long().to(device)
        end = [0] * B
        for i in range(L2-1):
            target = self.decoder_map2(fake) * math.sqrt(self.hidden_size)
            target_pe = self.pe(target)
            tgt_mask = self._generate_square_subsequent_mask(target_pe.shape[1]).to(device)
            seq_feat = self.decoder(target_pe, memory, tgt_mask=tgt_mask)

            pi = self.pred_pi(seq_feat[:, -1:])
            pi = F.softmax(pi, dim=-1)

            mu = self.pred_mu(seq_feat[:, -1:])
            mu = mu.view(B, 1, 20, 2)

            sigma = self.pred_sigma(seq_feat[:, -1:])
            sigma = torch.minimum(torch.tensor(500.).to(sigma.device), torch.exp(sigma))
            sigma = sigma.view(B, 1, 20, 2)

            rho = self.pred_rho(seq_feat[:, -1:])
            rho = torch.tanh(rho)

            p = sample_point((pi, mu, sigma, rho), device, greedy)

            label = self.pred_label2(seq_feat[:, -1:])
            label = label.argmax(-1)

            for j in range(B):
                if label[j] == 2:
                    end[j] = 1
                if end[j]:
                    label[j] = 2
                    p[j, :] = 0 
            label_tmp = F.one_hot(label, 4)
            tmp = torch.cat((p, label_tmp), -1)
            fake = torch.cat((fake, tmp), 1)
        return fake, seq_feat


class mergemodel(nn.Module):
    def __init__(self, config):
        super(mergemodel, self).__init__()
        self.image_branch = image_branch(config['image'])
        self.seq_branch = seq_branch(config['seq'])  # params_num: 63463237, self test. total_params = sum(param.numel() for param in transformer_main.parameters())

    def forward(self, img, seq, device):
        content, style, target = seq
        output_seq = self.seq_branch(content, style, target, device)

        content, style = img
        output_image = self.image_branch(content, style, output_seq['seq_feat'])

        return output_image, output_seq

    def generate(self, img, seq, device):
        with torch.no_grad():
            content, style, target = seq
            output_seq = self.seq_branch(content, style, target, device)

            output_seq_generate, _ = self.seq_branch.generate(content, style, target, device)
            output_seq_generate_greedy, seq_feat = self.seq_branch.generate(content, style, target, device, greedy=True)

            content, style = img
            output_img = self.image_branch(content, style, seq_feat)
        return output_img, output_seq, output_seq_generate, output_seq_generate_greedy


class SkeletonDecoder(nn.Module):
    """修正版：输出 [B, 255, 3]（x,y,visible），匹配 SkeletonLoss 要求"""
    def __init__(self, hidden_size=1024, output_n_points=255):
        super().__init__()
        self.output_n_points = output_n_points  # 固定为255（与--max_skeleton_len一致）
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 3)  # 输出x,y,visible（3维）
        )
        self.visible_sigmoid = nn.Sigmoid()  # 可见性映射到0-1

    def forward(self, x):
        # x: [B, 765, 1024]（encoder输出：3个参考样本×255点=765）
        B, seq_len, _ = x.shape
        # 高维特征→3维坐标
        coord = self.fc_layers(x)  # [B, 765, 3]
        # 截断到255个点（合并3个参考样本的特征，取前255个）
        coord = coord[:, :self.output_n_points, :]  # [B, 255, 3]
        # 可见性归一化（0-1）
        coord[..., 2] = self.visible_sigmoid(coord[..., 2])
        return coord  # [B, 255, 3]


if __name__ == '__main__':
    net1 = image_branch().cuda()
    style = torch.tensor(np.ones((16, 10, 3, 128, 128))).float().cuda()
    content = torch.tensor(np.ones((16, 3, 128, 128))).float().cuda()
    fake = net1(content, style)
    print(fake[0].shape)

    net2 = seq_branch().cuda()
    style = torch.tensor(np.ones((16, 10, 50, 5))).float().cuda()
    content = torch.tensor(np.ones((16, 50, 5))).float().cuda()
    target = torch.tensor(np.ones((16, 100, 5))).float().cuda()
    fake = net2(content, style, target)
    print(fake[0][0].shape)


