import torch
import torch.nn as nn
import torch.nn.functional as F

class SkeletonLoss(nn.Module):
    def __init__(self, 
                 weight_point: float = 1.0,
                 weight_chamfer: float = 5.0,
                 weight_structure: float = 2.0):
        super().__init__()
        # 损失权重（与模型初始化参数对应，支持外部调节）
        self.weight_point = weight_point
        self.weight_chamfer = weight_chamfer
        self.weight_structure = weight_structure

    def mask_mse_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """带掩码MSE：仅计算第三列=1的有效点，兼容少数空骨架（第三列=0）"""
        # pred/target: [B, N, 3]（B=批次，N=最大点数，第三列为有效掩码）
        valid_mask = (target[..., 2] == 1.0).float().unsqueeze(-1)  # [B, N, 1]
        # 仅对有效点计算坐标误差（前两列）
        coord_error = (pred[..., :2] - target[..., :2]) * valid_mask
        return torch.mean(coord_error ** 2)

    def chamfer_distance_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """倒角距离：解决不同样本点数差异（60/65/76等），保证整体形状匹配"""
        B = pred.shape[0]
        total_chamfer = 0.0

        for b in range(B):
            # 提取单批次有效点（过滤补零点和空骨架点）
            pred_valid = pred[b][pred[b][..., 2] == 1.0][..., :2]  # [Np, 2]
            target_valid = target[b][target[b][..., 2] == 1.0][..., :2]  # [Nt, 2]

            if len(pred_valid) == 0 or len(target_valid) == 0:
                continue  # 跳过空骨架样本

            # 双向最小距离计算（向量化操作，提升效率）
            dist_pred2target = torch.cdist(pred_valid.unsqueeze(0), target_valid.unsqueeze(0), p=2).squeeze(0)  # [Np, Nt]
            min_dist_pred = torch.min(dist_pred2target, dim=1)[0].mean()  # pred到target的平均最小距离

            dist_target2pred = torch.cdist(target_valid.unsqueeze(0), pred_valid.unsqueeze(0), p=2).squeeze(0)  # [Nt, Np]
            min_dist_target = torch.min(dist_target2pred, dim=1)[0].mean()  # target到pred的平均最小距离

            total_chamfer += (min_dist_pred + min_dist_target) / 2  # 双向平均

        return total_chamfer / B



    def structural_point_loss(self, pred: torch.Tensor, target: torch.Tensor, skeleton_mask: torch.Tensor) -> torch.Tensor:
        """结构点损失：强化端点/分支点精度（适配可选的结构掩码输入）"""
        if skeleton_mask is None:
            return torch.tensor(0.0, device=pred.device)

        # skeleton_mask: [B, N, 2]（第一列=端点，第二列=分支点）
        endpoint_mask = skeleton_mask[..., 0].float().unsqueeze(-1)  # [B, N, 1]
        junction_mask = skeleton_mask[..., 1].float().unsqueeze(-1)  # [B, N, 1]
        structure_mask = (endpoint_mask + junction_mask).clamp(0, 1)  # 合并结构点掩码

        # 仅对结构点计算坐标误差
        valid_mask = (target[..., 2] == 1.0).float().unsqueeze(-1)  # 结合有效点掩码
        total_mask = structure_mask * valid_mask

        if torch.sum(total_mask) == 0:
            return torch.tensor(0.0, device=pred.device)

        structure_error = (pred[..., :2] - target[..., :2]) * total_mask
        return torch.mean(structure_error ** 2)

    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor, 
                skeleton_mask: torch.Tensor = None) -> dict:
        """
        前向传播：匹配模型损失字典格式，兼容UNet前/后双阶段调用
        Args:
            pred: 预测骨架 [B, N, 3]
            target: 真实骨架 [B, N, 3]
            skeleton_mask: 结构点掩码 [B, N, 2]（可选，模型中可传None）
        Returns:
            损失字典：严格匹配modal_loss_dict的键名，无额外修改
        """
        # 计算各分项损失
        loss_point = self.mask_mse_loss(pred, target)
        loss_chamfer = self.chamfer_distance_loss(pred, target)
        loss_structure = self.structural_point_loss(pred, target, skeleton_mask)

        # 总损失（权重融合）
        total_loss = (
            self.weight_point * loss_point
            + self.weight_chamfer * loss_chamfer
            + self.weight_structure * loss_structure
        )

        # 返回格式：严格兼容模型的modal_loss_dict，不新增/删除键
        return {
            'skeleton_total_loss': total_loss,
            'skeleton_point_loss': loss_point,
            'skeleton_visibility_loss': torch.tensor(0.0, device=pred.device),  # 兼容原有结构（无用到时返回0）
            'skeleton_shape_loss': loss_chamfer  # 仅保留倒角距离作为形状损失
        }