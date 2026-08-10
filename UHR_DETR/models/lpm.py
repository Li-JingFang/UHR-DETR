import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalRankLoss(nn.Module):
    def __init__(self, margin=0.05, loss_weight=1.0):
        super().__init__()
        self.margin = margin
        self.loss_weight = loss_weight

    def forward(self, pred_map, gt_raw_map):
        """
        pred_map: [B, 1, H, W] (还原后的 Sqrt 预测值, 0~6.0)
        gt_raw_map: [B, 1, H, W] (原始 Sqrt GT, 0~6.0+)
        """
        # 4方向邻域 (上下左右)
        shifts = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        loss = torch.tensor(0.0, device=pred_map.device)
        valid_pairs = 0.0

        # 1. 动态掩码：只在有物体的地方(>1个)算 Rank，背景不参与
        mask_region = gt_raw_map >= 1.0

        for dy, dx in shifts:
            # 邻居的 GT 和 Pred
            gt_neighbor = torch.roll(gt_raw_map, shifts=(dy, dx), dims=(2, 3))
            pred_neighbor = torch.roll(pred_map, shifts=(dy, dx), dims=(2, 3))

            # 2. 找出"中心比邻居大"的位置 (峰值点)
            is_peak = (gt_raw_map > gt_neighbor) & mask_region

            if is_peak.sum() == 0:
                continue

            # 3. 强制 Pred 也要保持这个峰值关系，且至少高出 margin
            # Loss = ReLU( (Neighbor + margin) - Center )
            diff = (pred_neighbor[is_peak] + self.margin) - pred_map[is_peak]

            current_loss = F.relu(diff).mean()
            loss += current_loss
            valid_pairs += 1

        if valid_pairs > 0:
            loss = loss / valid_pairs
        loss = loss * self.loss_weight
        return loss