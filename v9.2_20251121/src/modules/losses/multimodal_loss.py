import torch
import torch.nn as nn
from typing import Dict, Optional

# 导入新的骨架损失函数
from .skeleton_loss import SkeletonLoss


class SVGModalLoss(nn.Module):
    """
    SVG模态损失函数
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, mask=None):
        assert pred.ndim == target.ndim, f"pred({pred.shape}) and target({target.shape}) should have same ndim"
        
        if mask is not None:
            # 应用掩码到预测和目标
            pred = pred * mask
            target = target * mask
        
        # 计算L1损失
        l1_loss = nn.functional.l1_loss(pred, target, reduction=self.reduction)
        
        loss_dict = {
            'svg_total_loss': l1_loss,
            # 以下为兼容原有损失字典结构的零值张量
            'visibility_loss': torch.tensor(0.0, device=pred.device),
            'shape_loss': torch.tensor(0.0, device=pred.device)
        }
        
        return loss_dict


class SkeletonModalLoss(nn.Module):
    """
    骨架模态损失函数
    使用新的组合损失函数，包含点坐标损失、Chamfer距离和结构点损失
    """
    def __init__(self,
                 weight_point: float = 1.0,
                 weight_chamfer: float = 2.5,
                 weight_structure: float = 1.5):
        super().__init__()
        # 初始化新的骨架损失函数
        self.skeleton_loss = SkeletonLoss(
            weight_point=weight_point,
            weight_chamfer=weight_chamfer,
            weight_structure=weight_structure
        )

    def forward(self,
                pred_skeleton: torch.Tensor,
                target_skeleton: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算骨架模态损失
        
        Args:
            pred_skeleton: 预测的骨架点序列，形状为[B, N, 3]
            target_skeleton: 目标骨架点序列，形状为[B, N, 3]
            mask: 可选的结构点掩码，形状为[B, N, 2]
        
        Returns:
            包含各种损失的字典
        """
        # 确保预测和目标形状相同
        assert pred_skeleton.shape == target_skeleton.shape, f"预测和目标形状不匹配: {pred_skeleton.shape} vs {target_skeleton.shape}"
        
        # 调用新的骨架损失函数
        return self.skeleton_loss(pred_skeleton, target_skeleton, mask)
        