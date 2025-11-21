import numpy as np
import torch
import torchvision 
import torch.nn.functional as F 
import torch.distributions as DIS


def mdnloss(real, fake, eval_mode=False):
    pi, mu, sigma, rho, fake_label = fake
    s1 = sigma[:, :, :, 0]
    s2 = sigma[:, :, :, 1]
    s1 = torch.clip(s1, 1e-6, 500.)
    s2 = torch.clip(s2, 1e-6, 500.)
    # s1 = torch.clip(s1, 0.5, 10.)  ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # s2 = torch.clip(s2, 0.5, 10.)
    mu1 = mu[:, :, :, 0]
    mu2 = mu[:, :, :, 1]
    x1 = real[:, :, 0:1]
    x2 = real[:, :, 1:2]
    # mask = 1 - real[:, :, 4]
    mask = real[:, :, -1]
    fake_label = fake_label.view(-1, fake_label.shape[-1])
    real_label = real[:, :, 2:].argmax(-1).view(-1).long()

    norm1 = x1 - mu1
    norm2 = x2 - mu2
    s1s2 = s1 * s2

    ### self modified
    # mask = real[:, :, -1]
    # dim = mu.shape[3]
    # z, s, norm = 0, 1, 1
    # for i in range(dim):
    #     s_i = sigma[:, :, :, i]
    #     s_i = torch.clip(s_i, 1e-6, 500.)
    #     mu_i = mu[:, :, :, i]
    #     x_i = real[:, :, i:i+1]
    #     norm_i = x_i - mu_i
    #     z += torch.square(norm_i / s_i)
    #     norm *= norm_i
    #     s *= s_i

    # print("mask.sum:", mask.sum().item(), "mask.unique:", torch.unique(mask))
    # print("pi sum:", pi.sum(-1).mean().item())
    # # print("sigma min/max:", sigma.min().item(), sigma.max().item())
    # print("s1 min/max:", s1.min().item(), s1.max().item())
    # print("rho min/max:", rho.min().item(), rho.max().item())
    # print("mu mean/std:", mu.mean().item(), mu.std().item())
    
    z = torch.square(norm1 / s1) + torch.square(norm2 / s2) - 2 * rho * norm1 * norm2 / s1s2
    # z = z - 2 * rho * norm / s
    neg_rho = torch.clip(1 - torch.square(rho), 1e-6, 1.0)
    result1 = torch.exp(-z / (2 * neg_rho))
    denom = 2 * np.pi * s1s2 * torch.sqrt(neg_rho)
    # denom = 2 * np.pi * s * torch.sqrt(neg_rho)
    result1 = result1 / denom
    result1 = (result1 * pi).sum(-1)
    # result1 = -torch.log(result1 + 1e-10)  ### no nll!!!!!
    result1 = 1. / (1 + result1)
    result1 = (result1 * mask).sum() / mask.sum().float()
    # print("pdf min/max:", result1.min().item(), result1.max().item(), "mean:", result1.mean().item())
    # # result1 = (result1 * mask).mean()
    # result1 = torch.clamp(result1, min=0)  ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    return result1#, result2


    # z = torch.square(norm1 / s1) + torch.square(norm2 / s2) - 2 * rho * norm1 * norm2 / s1s2
    # neg_rho = 1 - rho**2
    # log_pdf = -torch.log(2 * np.pi * s1 * s2) \
    #           - 0.5 * torch.log(neg_rho) \
    #           - z / (2 * neg_rho)
    # # log(Σ π * pdf) = logsumexp(logπ + log_pdf)
    # log_pi = torch.log(pi + 1e-9)
    # log_mix = torch.logsumexp(log_pi + log_pdf, dim=-1)
    # # NLL
    # nll = -log_mix
    # nll = (nll * mask).sum() / mask.sum().float()
    # print("nll min/max:", nll.min().item(), nll.max().item(), "mean:", nll.mean().item())
    # return nll
    






def diff(seq, device, gt=False):
    if gt:
        with torch.no_grad():
            point = seq[:, :, :2]
            mask = seq[:, :-1, 2]
            B, L, _ = point.shape
    else:
        pi, mu, sigma, rho, label = seq
        B, L, n, _ = mu.shape

        mix = DIS.Categorical(probs=pi).sample().unsqueeze(2).unsqueeze(2).repeat(1, 1, 1, 2)
        point = torch.gather(mu, 2, mix)

        mask = F.gumbel_softmax(label, tau=1, hard=True)
        mask = mask[:, :-1, 0]

    xs = torch.linspace(-1, 1, steps=128).to(device)
    ys = torch.linspace(-1, 1, steps=128).to(device)
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    x = x.reshape(1, -1, 1, 1).repeat(B, 1, 1, 1)
    y = y.reshape(1, -1, 1, 1).repeat(B, 1, 1, 1)
    point = point.view(B, 1, L, 2)
    mesh = torch.cat((x, y), -1)

    a = point[:, :, :-1]
    b = point[:, :, 1:]

    L1 = torch.norm(mesh-a, p=2, dim=-1)
    L2 = torch.norm(mesh-b, p=2, dim=-1)
    L3 = (mesh-a)[:, :, :, 0] * (mesh-b)[:, :, :, 1] - (mesh-a)[:, :, :, 1] * (mesh-b)[:, :, :, 0]
    L3 = L3 / (torch.norm(a-b, p=2, dim=-1)+1e-5)
    L3 = torch.abs(L3)
    mask1 = (b-a) * (mesh-a)
    mask1 = mask1.sum(-1) < 0
    mask2 = (a-b) * (mesh-b)
    mask2 = mask2.sum(-1) < 0
    mask1 = mask1.float()
    mask2 = mask2.float()

    D = L1 * mask1 + L2 * mask2 + L3 * (1-mask1) * (1-mask2)
    mask = mask.view(B, 1, -1).repeat(1, 128*128, 1).long()
    mask = (mask == 0)

    D.masked_fill_(mask, 1e5)
    D = torch.min(D, dim=-1)[0]
    D = D.view(B, 128, 128)
    D = 1 - torch.sigmoid(100*(D-4/128.))
    return D






### my bone-img cross-modal diff loss (chatgpt)
def sample_from_mdn(pi, mu, sigma, rho):
    """
    简单的 MDN 采样函数（2D Gaussian Mixture）
    pi: [B, L, K] mixture weights
    mu: [B, L, K, 2] means
    sigma: [B, L, K, 2] std
    rho: [B, L, K] correlation coefficient
    return: [B, L, 2] sampled points
    """
    B, L, K = pi.shape
    device = pi.device

    # 采样每个位置的成分
    cat_dist = torch.distributions.Categorical(pi)
    idx = cat_dist.sample()  # [B, L]

    # Gather 对应的 mu, sigma, rho
    mu_selected = mu[torch.arange(B)[:, None], torch.arange(L), idx]  # [B, L, 2]
    sigma_selected = sigma[torch.arange(B)[:, None], torch.arange(L), idx]  # [B, L, 2]
    rho_selected = rho[torch.arange(B)[:, None], torch.arange(L), idx]  # [B, L]

    # 采样二维高斯
    u1 = torch.randn(B, L, device=device)
    u2 = torch.randn(B, L, device=device)
    x = sigma_selected[..., 0] * u1
    y = sigma_selected[..., 1] * (rho_selected * u1 + torch.sqrt(1 - rho_selected**2) * u2)
    points = mu_selected + torch.stack([x, y], dim=-1)
    return points  # [B, L, 2]

def render_heatmap_normalized(points, H, W, sigma=0.03):
    """
    points: [B, L, 2], 归一化坐标 [0,1]
    H, W: 输出 heatmap 尺寸
    sigma: 高斯半径，归一化坐标 (相对于图像宽高)
    return: [B, H, W] heatmap
    """
    B, L, _ = points.shape
    device = points.device

    # 将归一化坐标映射到像素坐标
    px = points[..., 0] * (W - 1)
    py = points[..., 1] * (H - 1)

    # sigma 也映射到像素尺度
    sigma_x = sigma * W
    sigma_y = sigma * H

    xs = torch.arange(W, device=device).view(1, 1, 1, W)
    ys = torch.arange(H, device=device).view(1, 1, H, 1)

    px = px.view(B, L, 1, 1)
    py = py.view(B, L, 1, 1)

    dist2 = ((xs - px)**2) / (2 * sigma_x**2) + ((ys - py)**2) / (2 * sigma_y**2)
    heatmap = torch.exp(-dist2)

    # 合并 L 个点
    heatmap = heatmap.sum(1)  # [B, H, W]
    return heatmap

def cross_modal_loss_normalized(pi, mu, sigma_mdn, rho, target_image, H, W, sigma_heatmap=0.03, mse_weight=0.1):
    """
    MDN采样 -> heatmap渲染(归一化坐标 + 归一化sigma) -> BCE+MSE 跨模态损失
    pi, mu, sigma_mdn, rho: MDN 输出
        mu, sigma_mdn: [0,1] 归一化坐标
    target_image: [B,H,W] 二值图像
    H, W: 输出图像大小
    sigma_heatmap: heatmap高斯半径，归一化 [0,1]
    mse_weight: MSE权重
    """
    # 1. MDN 采样
    pred_points = sample_from_mdn(pi, mu, sigma_mdn, rho)  # [B, L, 2]

    # 2. 渲染 heatmap (归一化坐标 + sigma)
    pred_heatmap = render_heatmap_normalized(pred_points, H, W, sigma=sigma_heatmap)

    # 3. BCE + MSE 跨模态损失
    bce_loss = F.binary_cross_entropy_with_logits(pred_heatmap, target_image)
    mse_loss = F.mse_loss(torch.sigmoid(pred_heatmap), target_image)
    loss = bce_loss + mse_weight * mse_loss

    return loss, pred_heatmap

# 使用实例
# loss, pred_heatmap = cross_modal_loss_normalized(
#     pi, mu, sigma_mdn, rho, target_image, H=64, W=64, sigma_heatmap=0.03
# )