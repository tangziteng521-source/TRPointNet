import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils_knn import PointNetSetAbstractionMsg, PointNetFeaturePropagation


# 工具函数
def index_points(points, idx):
    """
    根据索引从点云中选取点
    points: [B, C, N]
    idx: [B, N, k]
    return: [B, C, N, k]
    """
    device = points.device
    B, C, N = points.shape
    _, _, k = idx.shape

    idx_expanded = idx.unsqueeze(1).expand(B, C, N, k)
    points_expanded = points.unsqueeze(-1).expand(B, C, N, k)
    new_points = torch.gather(points_expanded, 2, idx_expanded)

    return new_points


def knn(xyz, query_xyz, k):
    """
    K近邻搜索
    xyz: [B, 3, N]
    query_xyz: [B, 3, N]
    k: int
    return: [B, N, k]
    """
    inner = -2 * torch.matmul(query_xyz.transpose(2, 1), xyz)
    xx = torch.sum(query_xyz ** 2, dim=1, keepdim=True)
    yy = torch.sum(xyz ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - yy.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def estimate_normals(xyz, knn_indices, eps=1e-8):
    """
    通过PCA估计法向量 - 修复版本
    xyz: [B, 3, N]
    knn_indices: [B, N, k]
    return: [B, 3, N] 法向量
    """
    B, _, N = xyz.shape
    k = knn_indices.shape[-1]
    
    # 获取KNN邻域点
    knn_xyz = index_points(xyz, knn_indices)  # [B, 3, N, k]
    
    # 重新排列维度为 [B, N, k, 3] 以便计算协方差
    knn_xyz_permuted = knn_xyz.permute(0, 2, 3, 1)  # [B, N, k, 3]
    
    # 计算局部协方差矩阵
    neighbor_means = knn_xyz_permuted.mean(dim=2, keepdim=True)  # [B, N, 1, 3]
    centered = knn_xyz_permuted - neighbor_means  # [B, N, k, 3]
    
    # 计算协方差矩阵 [B, N, 3, 3]
    cov = torch.matmul(centered.transpose(2, 3), centered) / (k - 1)  # [B, N, 3, 3]
    
    # 重塑为 [B*N, 3, 3] 进行SVD
    cov_flat = cov.reshape(-1, 3, 3)  # [B*N, 3, 3]
    
    try:
        # 特征值分解（使用SVD）
        U, S, V = torch.svd(cov_flat)
        # 最小特征值对应的特征向量即为法向量
        normals_flat = U[:, :, 2]  # [B*N, 3]
    except:
        # 如果SVD失败，使用随机法向量作为fallback
        normals_flat = torch.randn(B * N, 3, device=xyz.device)
        normals_flat = F.normalize(normals_flat, p=2, dim=1)
    
    # 重塑回原始形状 [B, N, 3] -> [B, 3, N]
    normals = normals_flat.reshape(B, N, 3).permute(0, 2, 1)
    
    # 统一法向量方向（指向正z轴）
    flip_mask = (normals[:, 2, :] < 0).float().unsqueeze(1)
    normals = (1 - 2 * flip_mask) * normals
    
    return normals


class EnhancedTreeGeometricAttention(nn.Module):
    """增强的几何注意力模块 - 包含法向量估计"""

    def __init__(self, channels, k_neighbors=16, use_normals=True):
        super().__init__()
        self.k = k_neighbors
        self.use_normals = use_normals

        # 根据是否使用法向量调整输入通道
        input_channels = 8 if use_normals else 4
        
        # 增强的几何编码器
        self.geom_encoder = nn.Sequential(
            nn.Conv2d(input_channels, channels // 2, 1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )

        # 特征变换
        self.feat_transform = nn.Conv1d(channels, channels, 1)

    def compute_enhanced_geometric_features(self, xyz, knn_indices):
        """
        增强的几何特征计算，包含法向量
        """
        B, _, N = xyz.shape

        # 获取KNN点
        knn_xyz = index_points(xyz, knn_indices)  # [B, 3, N, k]
        center_xyz = xyz.unsqueeze(-1)  # [B, 3, N, 1]

        # 1. 相对坐标
        rel_pos = knn_xyz - center_xyz  # [B, 3, N, k]

        # 2. 距离特征
        distances = torch.norm(rel_pos, dim=1, keepdim=True)  # [B, 1, N, k]

        if self.use_normals:
            # 3. 估计法向量
            with torch.no_grad():  # 法向量估计不参与梯度计算
                normals = estimate_normals(xyz, knn_indices)  # [B, 3, N]
            
            # 4. 法向量差异特征
            knn_normals = index_points(normals, knn_indices)  # [B, 3, N, k]
            center_normals = normals.unsqueeze(-1)  # [B, 3, N, 1]
            normal_diffs = 1 - torch.abs(
                torch.sum(knn_normals * center_normals, dim=1, keepdim=True)
            )  # [B, 1, N, k]，值域[0,2]，0表示方向相同
            
            # 5. 拼接所有几何特征
            geom_features = torch.cat([rel_pos, distances, normal_diffs, knn_normals], dim=1)  # [B, 8, N, k]
        else:
            # 简化的几何特征（不使用法向量）
            geom_features = torch.cat([rel_pos, distances], dim=1)  # [B, 4, N, k]

        return geom_features

    def forward(self, feat, xyz):
        B, C, N = feat.shape

        # 1. KNN索引
        knn_indices = knn(xyz, xyz, k=self.k)

        # 2. 计算增强的几何描述子
        geom_features = self.compute_enhanced_geometric_features(xyz, knn_indices)

        # 3. 生成注意力权重
        attn_weights = self.geom_encoder(geom_features)  # [B, C, N, k]

        # 4. 聚合邻居特征
        knn_feat = index_points(feat, knn_indices)  # [B, C, N, k]
        aggregated_feat = (knn_feat * attn_weights).max(dim=-1)[0]  # [B, C, N]

        # 5. 残差连接
        output = self.feat_transform(feat) + aggregated_feat

        return F.relu(output)


class ImprovedHierarchicalFeatureFusion(nn.Module):
    """改进的层次特征融合模块 - 包含通道注意力"""

    def __init__(self, in_channels_list, out_channels=128):
        super(ImprovedHierarchicalFeatureFusion, self).__init__()

        self.out_channels = out_channels
        total_channels = sum(in_channels_list)

        # 通道注意力模块
        self.channel_attentions = nn.ModuleList()
        for in_channels in in_channels_list:
            self.channel_attentions.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Conv1d(in_channels, max(8, in_channels // 8), 1),
                    nn.ReLU(),
                    nn.Conv1d(max(8, in_channels // 8), in_channels, 1),
                    nn.Sigmoid()
                )
            )

        # 融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(total_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 1)
        )

    def forward(self, features_list):
        # 应用通道注意力
        target_size = features_list[0].shape[2]
        attended_features = []
        
        for i, feat in enumerate(features_list):
            # 通道注意力加权
            channel_weights = self.channel_attentions[i](feat)
            weighted_feat = feat * channel_weights
            
            # 上采样到目标分辨率
            if weighted_feat.shape[2] != target_size:
                # 对于1D数据，使用线性插值
                resized = F.interpolate(
                    weighted_feat, 
                    size=target_size, 
                    mode='linear',  # 对于1D数据使用'linear'
                    align_corners=False
                )
            else:
                resized = weighted_feat
            attended_features.append(resized)

        # 拼接所有特征
        fused = torch.cat(attended_features, dim=1)

        # 融合
        output = self.fusion_conv(fused)

        return output


class get_model(nn.Module):
    def __init__(self, num_classes, use_normals=True):
        super(get_model, self).__init__()
        self.use_normals = use_normals

        # 3层SA结构 - 平衡容量和效率
        # SA1: 下采样到1024点，中等通道数
        self.sa1 = PointNetSetAbstractionMsg(
            1024, [16, 32], [16, 32], 9, 
            [[24, 24, 48], [32, 32, 64]]
        )  # 输出: 48 + 64 = 112通道
        
        # SA2: 下采样到256点，增加通道数
        self.sa2 = PointNetSetAbstractionMsg(
            256, [16, 32], [16, 32], 112, 
            [[64, 64, 128], [64, 96, 128]]
        )  # 输出: 128 + 128 = 256通道
        
        # SA3: 下采样到64点，进一步增加通道数
        self.sa3 = PointNetSetAbstractionMsg(
            64, [32, 64], [32, 64], 256,
            [[128, 128, 256], [128, 192, 256]]
        )  # 输出: 256 + 256 = 512通道

        # 增强的几何注意力模块
        self.geom_attn1 = EnhancedTreeGeometricAttention(channels=112, k_neighbors=16, use_normals=use_normals)
        self.geom_attn2 = EnhancedTreeGeometricAttention(channels=256, k_neighbors=12, use_normals=use_normals)
        self.geom_attn3 = EnhancedTreeGeometricAttention(channels=512, k_neighbors=8, use_normals=use_normals)

        # 改进的层次特征融合模块 - 融合4个层次的特征
        self.hierarchical_fusion = ImprovedHierarchicalFeatureFusion(
            [9, 112, 256, 512], out_channels=256
        )

        # FP层 - 调整通道数以匹配新的SA层输出
        self.fp3 = PointNetFeaturePropagation(256 + 512, [512, 256])  # 融合SA2和SA3
        self.fp2 = PointNetFeaturePropagation(112 + 256, [256, 128])  # 融合SA1和SA2
        self.fp1 = PointNetFeaturePropagation(128 + 9, [128, 128, 128])  # 融合原始和SA1

        # 输出层
        self.conv1 = nn.Conv1d(128 + 256, 256, 1)  # 融合FP输出和层次特征
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)  # 稍微降低dropout率
        self.conv2 = nn.Conv1d(256, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        # 3层SA编码 + 几何注意力
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # [B, 112, 1024]
        l1_points_attn = self.geom_attn1(l1_points, l1_xyz)
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points_attn)  # [B, 256, 256]
        l2_points_attn = self.geom_attn2(l2_points, l2_xyz)
        
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points_attn)  # [B, 512, 64]
        l3_points_attn = self.geom_attn3(l3_points, l3_xyz)

        # 3层FP解码
        l2_points_fp = self.fp3(l2_xyz, l3_xyz, l2_points_attn, l3_points_attn)  # [B, 256, 256]
        l1_points_fp = self.fp2(l1_xyz, l2_xyz, l1_points_attn, l2_points_fp)    # [B, 128, 1024]
        l0_points_fp = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points_fp)         # [B, 128, N]

        # 层次特征融合（4个层次）
        features_list = [
            l0_points,        # 原始特征 [B, 9, N]
            l1_points_attn,   # SA1+注意力 [B, 112, 1024] 
            l2_points_attn,   # SA2+注意力 [B, 256, 256]
            l3_points_attn    # SA3+注意力 [B, 512, 64]
        ]

        hierarchical_features = self.hierarchical_fusion(features_list)  # [B, 256, N]

        # 融合FP特征和层次特征
        final_features = torch.cat([l0_points_fp, hierarchical_features], dim=1)  # [B, 384, N]

        # 输出层
        x = F.relu(self.bn1(self.conv1(final_features)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)
        return total_loss


if __name__ == '__main__':
    import torch

    # 测试模型
    model = get_model(3, use_normals=True)
    xyz = torch.rand(6, 9, 2048)
    output, _ = model(xyz)
    print(f"输出形状: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    # 测试不使用方法向量的版本
    model_simple = get_model(3, use_normals=False)
    print(f"简化版参数量: {sum(p.numel() for p in model_simple.parameters() if p.requires_grad) / 1e6:.2f}M")