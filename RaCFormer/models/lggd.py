# ==============================================================================
# LGGD模块 (Learnable Gaussian-Geometry Densification)
# 功能: 基于高斯溅射原理的可微分雷达点云稠密化模块
# 特点: 将稀疏雷达点云转换为密集BEV特征图
# 接入位置: 替换或增强现有的雷达编码器
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from typing import Tuple, Optional


class PointEncoder(nn.Module):
    """
    点云特征编码器
    
    使用MLP将原始雷达点云编码为高维特征
    
    Architecture:
        Input (B, N, in_channels) -> MLP -> Output (B, N, hidden_dim)
    """
    
    def __init__(self, in_channels: int = 6, hidden_dim: int = 64, num_layers: int = 2):
        """
        初始化点云编码器
        
        Args:
            in_channels: 输入通道数 (默认6: x, y, z, vx, vy, rcs)
            hidden_dim: 隐藏层维度
            num_layers: MLP层数
        """
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
        
        self.mlp = nn.Sequential(*layers)
        self.hidden_dim = hidden_dim
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            points: 输入点云 [B, N, in_channels]
            
        Returns:
            point_feats: 点云特征 [B, N, hidden_dim]
        """
        B, N, C = points.shape  # [B, N, in_channels]
        
        # Reshape for BatchNorm1d: [B*N, C] -> MLP -> [B*N, hidden_dim]
        x = points.reshape(B * N, C)  # [B*N, in_channels]
        
        # Apply MLP with BatchNorm
        for layer in self.mlp:
            if isinstance(layer, nn.BatchNorm1d):
                x = layer(x)  # BatchNorm1d expects [N, C]
            else:
                x = layer(x)
        
        # Reshape back: [B*N, hidden_dim] -> [B, N, hidden_dim]
        point_feats = x.reshape(B, N, self.hidden_dim)  # [B, N, hidden_dim]
        
        return point_feats


class GeometryHeads(nn.Module):
    """
    几何参数预测头
    
    从点云特征预测高斯溅射所需的几何参数:
    - offset: 位置偏移 Δμ ∈ R³
    - scale: 尺度 s ∈ R² (BEV x, y)
    - rotation: 旋转角度 θ
    - opacity: 不透明度 α ∈ [0, 1]
    """
    
    def __init__(self, hidden_dim: int = 64, offset_limit: float = 2.0):
        """
        初始化几何预测头
        
        Args:
            hidden_dim: 输入特征维度
            offset_limit: 位置偏移的最大范围 (米)
        """
        super().__init__()
        
        # 偏移预测头: 输出 Δμ ∈ R³ (x, y, z偏移)
        self.offset_head = nn.Linear(hidden_dim, 3)  # [B, N, 3]
        
        # 尺度预测头: 输出 s ∈ R² (BEV平面的x, y尺度)
        self.scale_head = nn.Linear(hidden_dim, 2)  # [B, N, 2]
        
        # 旋转预测头: 输出 (cos θ, sin θ)
        self.rotation_head = nn.Linear(hidden_dim, 2)  # [B, N, 2]
        
        # 不透明度预测头: 输出 α ∈ [0, 1]
        self.opacity_head = nn.Linear(hidden_dim, 1)  # [B, N, 1]
        
        # 可学习的偏移限制
        self.register_buffer('offset_limit', torch.tensor(offset_limit))
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重以确保稳定训练"""
        # 偏移头：小的初始偏移
        nn.init.xavier_uniform_(self.offset_head.weight, gain=0.1)
        nn.init.zeros_(self.offset_head.bias)
        
        # 尺度头：初始化为较小的值，exp后接近1
        nn.init.xavier_uniform_(self.scale_head.weight, gain=0.1)
        nn.init.zeros_(self.scale_head.bias)
        
        # 旋转头：初始化为接近(1, 0)即无旋转
        nn.init.xavier_uniform_(self.rotation_head.weight, gain=0.1)
        nn.init.constant_(self.rotation_head.bias[0], 1.0)  # cos
        nn.init.constant_(self.rotation_head.bias[1], 0.0)  # sin
        
        # 不透明度头：初始化为sigmoid的0附近，即α≈0.5
        nn.init.xavier_uniform_(self.opacity_head.weight, gain=0.1)
        nn.init.zeros_(self.opacity_head.bias)
    
    def forward(self, point_feats: torch.Tensor) -> dict:
        """
        前向传播
        
        Args:
            point_feats: 点云特征 [B, N, hidden_dim]
            
        Returns:
            geometry_params: 包含以下字段的字典:
                - offset: [B, N, 3] 位置偏移，经过tanh约束
                - scale: [B, N, 2] 尺度，经过exp确保正值
                - rotation: [B, N, 2] 旋转 (cos θ, sin θ)，归一化
                - opacity: [B, N, 1] 不透明度，经过sigmoid约束到[0,1]
        """
        # 偏移预测: tanh * offset_limit 约束到 [-limit, limit]
        offset = torch.tanh(self.offset_head(point_feats)) * self.offset_limit  # [B, N, 3]
        
        # 尺度预测: exp 确保正值, 添加最小值防止过小
        scale = torch.exp(self.scale_head(point_feats)).clamp(min=0.1, max=10.0)  # [B, N, 2]
        
        # 旋转预测: 归一化到单位圆
        rotation_raw = self.rotation_head(point_feats)  # [B, N, 2]
        rotation_norm = F.normalize(rotation_raw, p=2, dim=-1)  # [B, N, 2] 单位向量
        
        # 不透明度预测: sigmoid 约束到 [0, 1]
        opacity = torch.sigmoid(self.opacity_head(point_feats))  # [B, N, 1]
        
        return {
            'offset': offset,      # [B, N, 3]
            'scale': scale,        # [B, N, 2]
            'rotation': rotation_norm,  # [B, N, 2]
            'opacity': opacity     # [B, N, 1]
        }


class DifferentiableSplatting(nn.Module):
    """
    可微分溅射模块
    
    将点云特征通过高斯溅射方式投射到BEV网格上
    使用纯PyTorch实现，确保可移植性和可微分性
    
    核心操作:
    1. 将连续坐标映射到离散网格索引
    2. 使用scatter_add累积特征
    3. 可选的高斯加权
    """
    
    def __init__(
        self,
        bev_size: Tuple[int, int] = (128, 128),
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        use_gaussian_weight: bool = True,
        sigma_scale: float = 1.0
    ):
        """
        初始化可微分溅射模块
        
        Args:
            bev_size: BEV网格尺寸 (H, W)
            pc_range: 点云范围 (x_min, y_min, z_min, x_max, y_max, z_max)
            use_gaussian_weight: 是否使用高斯权重
            sigma_scale: 高斯sigma的缩放因子
        """
        super().__init__()
        
        self.bev_h, self.bev_w = bev_size
        self.pc_range = pc_range
        self.use_gaussian_weight = use_gaussian_weight
        self.sigma_scale = sigma_scale
        
        # 计算网格分辨率
        self.x_min, self.y_min, self.z_min = pc_range[0], pc_range[1], pc_range[2]
        self.x_max, self.y_max, self.z_max = pc_range[3], pc_range[4], pc_range[5]
        
        self.x_res = (self.x_max - self.x_min) / self.bev_w
        self.y_res = (self.y_max - self.y_min) / self.bev_h
    
    def world_to_grid(self, points_xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将世界坐标转换为网格索引
        
        Args:
            points_xy: 点的XY坐标 [B, N, 2]
            
        Returns:
            grid_u: X方向网格索引 [B, N]
            grid_v: Y方向网格索引 [B, N]
        """
        # 计算网格索引
        u = (points_xy[..., 0] - self.x_min) / self.x_res  # [B, N]
        v = (points_xy[..., 1] - self.y_min) / self.y_res  # [B, N]
        
        return u, v
    
    def compute_gaussian_weights(
        self,
        grid_u: torch.Tensor,
        grid_v: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor
    ) -> torch.Tensor:
        """
        计算高斯权重（简化版本，用于特征加权）
        
        Args:
            grid_u: X方向网格索引 [B, N]
            grid_v: Y方向网格索引 [B, N]
            scale: 尺度参数 [B, N, 2]
            rotation: 旋转参数 [B, N, 2] (cos θ, sin θ)
            
        Returns:
            weights: 高斯权重 [B, N]
        """
        # 简化处理：使用尺度的倒数作为基础权重
        # 较小的尺度 -> 较集中的高斯 -> 较高的权重
        sigma_x = scale[..., 0] * self.sigma_scale  # [B, N]
        sigma_y = scale[..., 1] * self.sigma_scale  # [B, N]
        
        # 高斯权重（在中心点处的值）
        # 对于标准高斯，中心点处的值为 1/(2π σx σy)
        # 为了数值稳定，我们使用归一化的权重
        weights = 1.0 / (sigma_x * sigma_y + 1e-6)  # [B, N]
        
        # 归一化权重
        weights = weights / (weights.max(dim=-1, keepdim=True)[0] + 1e-6)
        
        return weights
    
    def forward(
        self,
        points_xyz: torch.Tensor,
        point_feats: torch.Tensor,
        geometry_params: dict,
        valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：将点云特征溅射到BEV网格
        
        Args:
            points_xyz: 原始点云坐标 [B, N, 3]
            point_feats: 点云特征 [B, N, C]
            geometry_params: 几何参数字典
            valid_mask: 有效点掩码 [B, N], None表示全部有效
            
        Returns:
            bev_features: BEV特征图 [B, C, H, W]
            bev_counts: 每个格子的点数 [B, 1, H, W]
        """
        B, N, C = point_feats.shape
        device = point_feats.device
        dtype = point_feats.dtype
        
        # ========== Step 1: 应用位置偏移 ==========
        offset = geometry_params['offset']  # [B, N, 3]
        points_refined = points_xyz + offset  # [B, N, 3]
        
        # ========== Step 2: 计算网格索引 ==========
        grid_u, grid_v = self.world_to_grid(points_refined[..., :2])  # [B, N] each
        
        # 转换为整数索引
        grid_u_int = grid_u.long()  # [B, N]
        grid_v_int = grid_v.long()  # [B, N]
        
        # ========== Step 3: 创建有效掩码 ==========
        # 检查点是否在BEV边界内
        valid_x = (grid_u_int >= 0) & (grid_u_int < self.bev_w)  # [B, N]
        valid_y = (grid_v_int >= 0) & (grid_v_int < self.bev_h)  # [B, N]
        boundary_mask = valid_x & valid_y  # [B, N]
        
        # 合并外部提供的掩码
        if valid_mask is not None:
            final_mask = boundary_mask & valid_mask  # [B, N]
        else:
            final_mask = boundary_mask  # [B, N]
        
        # ========== Step 4: 计算特征权重 ==========
        opacity = geometry_params['opacity']  # [B, N, 1]
        scale = geometry_params['scale']  # [B, N, 2]
        rotation = geometry_params['rotation']  # [B, N, 2]
        
        if self.use_gaussian_weight:
            gauss_weights = self.compute_gaussian_weights(grid_u, grid_v, scale, rotation)  # [B, N]
            weights = opacity.squeeze(-1) * gauss_weights  # [B, N]
        else:
            weights = opacity.squeeze(-1)  # [B, N]
        
        # 对特征加权
        weighted_feats = point_feats * weights.unsqueeze(-1)  # [B, N, C]
        
        # ========== Step 5: 使用scatter_add累积特征 ==========
        # 初始化BEV画布
        bev_features = torch.zeros(B, C, self.bev_h, self.bev_w, device=device, dtype=dtype)
        bev_counts = torch.zeros(B, 1, self.bev_h, self.bev_w, device=device, dtype=dtype)
        
        # 逐batch处理（scatter_add不支持batch维度）
        for b in range(B):
            mask_b = final_mask[b]  # [N]
            
            if mask_b.sum() == 0:
                continue
            
            # 获取有效点的索引和特征
            valid_indices = mask_b.nonzero(as_tuple=True)[0]  # [M]
            valid_u = grid_u_int[b, valid_indices]  # [M]
            valid_v = grid_v_int[b, valid_indices]  # [M]
            valid_feats = weighted_feats[b, valid_indices]  # [M, C]
            valid_weights = weights[b, valid_indices]  # [M]
            
            # 计算线性索引: idx = v * W + u
            linear_idx = valid_v * self.bev_w + valid_u  # [M]
            
            # 累积特征: [M, C] -> [H*W, C]
            flat_bev = bev_features[b].view(C, -1)  # [C, H*W]
            flat_counts = bev_counts[b].view(1, -1)  # [1, H*W]
            
            # 使用index_add_累积
            # 对每个通道进行累积
            for c_idx in range(C):
                flat_bev[c_idx].index_add_(0, linear_idx, valid_feats[:, c_idx])
            
            # 累积计数
            ones = torch.ones(valid_indices.shape[0], device=device, dtype=dtype)
            flat_counts[0].index_add_(0, linear_idx, ones)
        
        # ========== Step 6: 归一化（可选）==========
        # 使用计数归一化，避免除零
        # bev_features = bev_features / (bev_counts + 1e-6)
        
        return bev_features, bev_counts


class FeatureSmoother(nn.Module):
    """
    特征平滑模块
    
    对BEV特征图进行卷积平滑，模拟高斯分布的扩散效果
    填充稀疏区域的空洞
    """
    
    def __init__(self, in_channels: int, out_channels: int = None, kernel_size: int = 3):
        """
        初始化特征平滑模块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数，None则与输入相同
            kernel_size: 卷积核大小
        """
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels
        
        padding = kernel_size // 2
        
        self.smoother = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            bev_features: BEV特征图 [B, C, H, W]
            
        Returns:
            smoothed_features: 平滑后的特征图 [B, C_out, H, W]
        """
        return self.smoother(bev_features)


class LGGD(BaseModule):
    """
    LGGD (Learnable Gaussian-Geometry Densification)
    
    可学习的高斯几何稠密化模块
    
    将稀疏雷达点云通过高斯溅射原理转换为密集BEV特征图
    
    Pipeline:
    1. PointEncoder: 点云 -> 高维特征
    2. GeometryHeads: 预测高斯参数 (offset, scale, rotation, opacity)
    3. DifferentiableSplatting: 可微分溅射到BEV网格
    4. FeatureSmoother: 卷积平滑填充空洞
    
    输入:
        - points: 雷达点云 [B, N, C] 或列表形式
        
    输出:
        - bev_features: BEV特征图 [B, out_channels, H, W]
    """
    
    def __init__(
        self,
        in_channels: int = 6,
        hidden_dim: int = 64,
        out_channels: int = 64,
        bev_size: Tuple[int, int] = (128, 128),
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        offset_limit: float = 2.0,
        use_gaussian_weight: bool = True,
        sigma_scale: float = 1.0,
        num_encoder_layers: int = 2,
        smoother_kernel_size: int = 3,
        enabled: bool = True,
        init_cfg: dict = None
    ):
        """
        初始化LGGD模块
        
        Args:
            in_channels: 输入点云通道数 (默认6: x, y, z, vx, vy, rcs)
            hidden_dim: 隐藏层维度
            out_channels: 输出BEV特征通道数
            bev_size: BEV网格尺寸 (H, W)
            pc_range: 点云范围 (x_min, y_min, z_min, x_max, y_max, z_max)
            offset_limit: 位置偏移最大范围 (米)
            use_gaussian_weight: 是否使用高斯权重
            sigma_scale: 高斯sigma缩放因子
            num_encoder_layers: 编码器MLP层数
            smoother_kernel_size: 平滑卷积核大小
            enabled: 是否启用模块
        """
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.bev_size = bev_size
        self.pc_range = pc_range
        self.enabled = enabled
        
        # ========== 子模块初始化 ==========
        
        # 1. 点云编码器
        self.point_encoder = PointEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers
        )
        
        # 2. 几何参数预测头
        self.geometry_heads = GeometryHeads(
            hidden_dim=hidden_dim,
            offset_limit=offset_limit
        )
        
        # 3. 可微分溅射模块
        self.splatting = DifferentiableSplatting(
            bev_size=bev_size,
            pc_range=pc_range,
            use_gaussian_weight=use_gaussian_weight,
            sigma_scale=sigma_scale
        )
        
        # 4. 特征平滑模块
        self.smoother = FeatureSmoother(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=smoother_kernel_size
        )
        
        # 计算BEV网格参数
        self.bev_h, self.bev_w = bev_size
    
    def preprocess_points(self, points) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预处理点云数据
        
        将不同格式的点云转换为统一的 [B, N, C] 格式
        
        Args:
            points: 点云数据，可以是:
                - Tensor [B, N, C]
                - List of Tensors [[N1, C], [N2, C], ...]
                
        Returns:
            points_tensor: [B, N, C]
            valid_mask: [B, N] 有效点掩码
        """
        if isinstance(points, torch.Tensor):
            if points.dim() == 3:
                # 已经是 [B, N, C] 格式
                B, N, C = points.shape
                # 创建有效掩码（非零点）
                valid_mask = (points.abs().sum(dim=-1) > 1e-6)  # [B, N]
                return points, valid_mask
            elif points.dim() == 2:
                # [N, C] 格式，添加batch维度
                points = points.unsqueeze(0)  # [1, N, C]
                valid_mask = (points.abs().sum(dim=-1) > 1e-6)  # [1, N]
                return points, valid_mask
        
        elif isinstance(points, (list, tuple)):
            # List of Tensors
            device = points[0].device
            dtype = points[0].dtype
            
            # 找到最大点数
            max_points = max(p.shape[0] for p in points)
            B = len(points)
            C = points[0].shape[-1]
            
            # 创建填充后的张量
            points_padded = torch.zeros(B, max_points, C, device=device, dtype=dtype)
            valid_mask = torch.zeros(B, max_points, device=device, dtype=torch.bool)
            
            for i, pts in enumerate(points):
                n = pts.shape[0]
                points_padded[i, :n] = pts
                valid_mask[i, :n] = True
            
            return points_padded, valid_mask
        
        else:
            raise TypeError(f"Unsupported points type: {type(points)}")
    
    def forward(
        self,
        points,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        LGGD前向传播
        
        Args:
            points: 雷达点云，支持多种格式:
                - Tensor [B, N, C]: 批量点云
                - List[Tensor]: 每个元素是 [N_i, C]
            return_intermediate: 是否返回中间结果
            
        Returns:
            bev_features: BEV特征图 [B, out_channels, H, W]
            (可选) intermediate: 中间结果字典
        """
        if not self.enabled:
            # 如果模块禁用，返回零初始化的BEV特征
            if isinstance(points, torch.Tensor):
                B = points.shape[0]
                device = points.device
                dtype = points.dtype
            else:
                B = len(points)
                device = points[0].device
                dtype = points[0].dtype
            
            bev_features = torch.zeros(B, self.out_channels, self.bev_h, self.bev_w,
                                       device=device, dtype=dtype)
            if return_intermediate:
                return bev_features, {}
            return bev_features
        
        # ========== Step 1: 预处理点云 ==========
        points_tensor, valid_mask = self.preprocess_points(points)
        # points_tensor: [B, N, C]
        # valid_mask: [B, N]
        
        B, N, C = points_tensor.shape
        
        # 确保通道数正确
        if C > self.in_channels:
            points_input = points_tensor[..., :self.in_channels]  # [B, N, in_channels]
        else:
            points_input = points_tensor  # [B, N, C]
        
        # 提取XYZ坐标
        points_xyz = points_tensor[..., :3]  # [B, N, 3]
        
        # ========== Step 2: 点云特征编码 ==========
        point_feats = self.point_encoder(points_input)  # [B, N, hidden_dim]
        
        # ========== Step 3: 预测几何参数 ==========
        geometry_params = self.geometry_heads(point_feats)
        # geometry_params包含: offset [B, N, 3], scale [B, N, 2], 
        #                     rotation [B, N, 2], opacity [B, N, 1]
        
        # ========== Step 4: 可微分溅射 ==========
        bev_raw, bev_counts = self.splatting(
            points_xyz, point_feats, geometry_params, valid_mask
        )
        # bev_raw: [B, hidden_dim, H, W]
        # bev_counts: [B, 1, H, W]
        
        # ========== Step 5: 特征平滑 ==========
        bev_features = self.smoother(bev_raw)  # [B, out_channels, H, W]
        
        if return_intermediate:
            intermediate = {
                'points_xyz': points_xyz,
                'points_refined': points_xyz + geometry_params['offset'],
                'point_feats': point_feats,
                'geometry_params': geometry_params,
                'bev_raw': bev_raw,
                'bev_counts': bev_counts,
                'valid_mask': valid_mask
            }
            return bev_features, intermediate
        
        return bev_features


class LGGDWrapper(BaseModule):
    """
    LGGD包装器
    
    用于与RaCFormer主模型集成的包装器类
    处理体素化输入和BEV特征生成
    """
    
    def __init__(
        self,
        lggd_cfg: dict = None,
        init_cfg: dict = None
    ):
        """
        初始化LGGD包装器
        
        Args:
            lggd_cfg: LGGD模块配置
        """
        super().__init__(init_cfg=init_cfg)
        
        if lggd_cfg is None:
            lggd_cfg = dict(
                in_channels=6,
                hidden_dim=64,
                out_channels=64,
                bev_size=(128, 128),
                pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                offset_limit=2.0,
                use_gaussian_weight=True,
                sigma_scale=1.0,
                num_encoder_layers=2,
                smoother_kernel_size=3,
                enabled=True
            )
        
        self.lggd = LGGD(**lggd_cfg)
        self.enabled = lggd_cfg.get('enabled', True)
        self.out_channels = lggd_cfg.get('out_channels', 64)
    
    def forward(self, radar_points) -> torch.Tensor:
        """
        前向传播
        
        Args:
            radar_points: 雷达点云列表 [B] x [N_i, C]
            
        Returns:
            bev_features: BEV特征图 [B, out_channels, H, W]
        """
        return self.lggd(radar_points)


# ==============================================================================
# 配置构建函数
# ==============================================================================

def build_radar_encoder(cfg: dict):
    """
    根据配置构建雷达编码器
    
    支持三种模式:
    - 'original': 原始的PillarFeatureNet
    - 'rhgm': RHGM增强模块
    - 'radarbevnet': RadarBEVNet双流编码器
    - 'lggd': LGGD高斯溅射稠密化模块
    
    Args:
        cfg: 配置字典，需包含 'type' 字段
        
    Returns:
        encoder: 构建的编码器模块
        
    Example:
        >>> cfg = dict(
        ...     type='lggd',
        ...     in_channels=6,
        ...     hidden_dim=64,
        ...     out_channels=64,
        ...     bev_size=(128, 128),
        ...     pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        ... )
        >>> encoder = build_radar_encoder(cfg)
    """
    encoder_type = cfg.pop('type', 'original').lower()
    
    if encoder_type == 'lggd':
        return LGGD(**cfg)
    elif encoder_type == 'lggd_wrapper':
        return LGGDWrapper(lggd_cfg=cfg)
    elif encoder_type == 'radarbevnet':
        from .radar_bev_net import RadarBEVNet
        return RadarBEVNet(**cfg)
    elif encoder_type == 'rhgm':
        from .rhgm import RHGM
        return RHGM(**cfg)
    else:
        raise ValueError(f"Unknown radar encoder type: {encoder_type}. "
                        f"Supported types: ['original', 'rhgm', 'radarbevnet', 'lggd']")


# ==============================================================================
# 测试代码
# ==============================================================================

if __name__ == "__main__":
    # 测试LGGD模块
    print("Testing LGGD Module...")
    
    # 创建模块
    lggd = LGGD(
        in_channels=6,
        hidden_dim=64,
        out_channels=64,
        bev_size=(128, 128),
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    )
    
    # 创建测试数据
    B, N, C = 2, 1000, 6
    points = torch.randn(B, N, C)
    
    # 设置点云在有效范围内
    points[..., 0] = points[..., 0] * 50  # x
    points[..., 1] = points[..., 1] * 50  # y
    points[..., 2] = points[..., 2] * 2   # z
    
    # 前向传播
    bev_features, intermediate = lggd(points, return_intermediate=True)
    
    print(f"Input points shape: {points.shape}")
    print(f"Output BEV features shape: {bev_features.shape}")
    print(f"BEV raw shape: {intermediate['bev_raw'].shape}")
    print(f"BEV counts max: {intermediate['bev_counts'].max().item()}")
    print(f"Geometry params:")
    for k, v in intermediate['geometry_params'].items():
        print(f"  {k}: {v.shape}")
    
    print("\nLGGD Module Test Passed!")

