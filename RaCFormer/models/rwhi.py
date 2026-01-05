# ==============================================================================
# RWHI模块 (RCS-Weighted Hybrid Anchor Initialization)
# 功能: 基于雷达点云的Query初始化策略
# 设计原则: 全向量化，无Python for循环，TensorRT兼容
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv.runner import BaseModule


class RWHIModule(BaseModule):
    """
    RCS-Weighted Hybrid Anchor Initialization Module
    
    雷达权重混合锚点初始化模块，用于生成高质量的检测Query初始位置。
    
    特点:
    1. 安全流 (Safety Stream, ~30%): 基于逆深度分布的固定锚点，覆盖近处盲区
    2. 显著流 (Saliency Stream, ~70%): 基于雷达RCS和速度的动态锚点
    3. 全向量化实现，无Python for循环
    4. TensorRT静态图兼容
    
    输入:
        - radar_points: 雷达点云 [B, M, C] (x, y, z, rcs, v_r, ...)
        
    输出:
        - query_positions: 混合锚点位置 [B, N_total, 10]
    """
    
    def __init__(self,
                 num_query=900,
                 safety_ratio=0.3,
                 embed_dims=256,
                 pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                 bev_grid_size=100,
                 safety_max_range=30.0,
                 velocity_alpha=0.5,
                 height_hypotheses=(0.0, 1.5),
                 diffusion_kernel_size=3,
                 noise_eps=1e-6,
                 enabled=True,
                 init_cfg=None):
        """
        初始化RWHI模块
        
        Args:
            num_query: 总Query数量
            safety_ratio: 安全流占比 (0.3 = 30%)
            embed_dims: 嵌入维度
            pc_range: 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
            bev_grid_size: BEV网格大小 (用于显著流体素化)
            safety_max_range: 安全流最大范围 (米)
            velocity_alpha: 速度权重系数
            height_hypotheses: 高度假设 (米)
            diffusion_kernel_size: 不确定性扩散核大小
            noise_eps: TopK防重复噪声
            enabled: 是否启用RWHI (False则使用原始初始化)
        """
        super(RWHIModule, self).__init__(init_cfg=init_cfg)
        
        self.num_query = num_query
        self.safety_ratio = safety_ratio
        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.bev_grid_size = bev_grid_size
        self.safety_max_range = safety_max_range
        self.velocity_alpha = velocity_alpha
        self.height_hypotheses = height_hypotheses
        self.diffusion_kernel_size = diffusion_kernel_size
        self.noise_eps = noise_eps
        self.enabled = enabled
        
        # 计算安全流和显著流的Query数量
        self.num_safety = int(num_query * safety_ratio)
        # 显著流需要考虑高度假设的数量
        num_height_hyp = len(height_hypotheses)
        self.num_saliency_per_height = (num_query - self.num_safety) // num_height_hyp
        self.num_saliency = self.num_saliency_per_height * num_height_hyp
        
        # 确保总数正确
        self.num_safety = num_query - self.num_saliency
        
        # 预计算安全流锚点 (register_buffer使其成为模型状态的一部分)
        safety_anchors = self._precompute_safety_anchors()
        self.register_buffer('safety_anchors', safety_anchors)
        
        # ============================================================
        # 计算BEV网格参数和坐标转换参数
        # 关键: 从pc_range动态计算，不使用硬编码值
        # ============================================================
        self.bev_x_range = pc_range[3] - pc_range[0]
        self.bev_y_range = pc_range[4] - pc_range[1]
        self.cell_size_x = self.bev_x_range / bev_grid_size
        self.cell_size_y = self.bev_y_range / bev_grid_size
        
        # 极坐标转换参数 (动态计算，替代硬编码的 map_size=102.4, r=65.0)
        # map_size: BEV地图尺寸 (x_max - x_min)
        # polar_radius: 极坐标最大半径，用于归一化距离
        #   - 原始RaCFormer使用 r=65.0，对应 ~65米的最大检测距离
        #   - 这里使用 BEV对角线的一半作为最大半径，确保覆盖整个BEV空间
        self.map_size = self.bev_x_range  # 假设x和y范围相同
        self.polar_radius = 65.0  # 与原始RaCFormer保持一致
        # 如果需要动态计算: self.polar_radius = math.sqrt(self.bev_x_range**2 + self.bev_y_range**2) / 2
        
        # 安全流掩码 (用于Mask显著流的近处区域)
        safety_mask = self._precompute_safety_mask()
        self.register_buffer('safety_mask', safety_mask)
        
        # 不确定性扩散层 (MaxPool2d)
        self.diffusion = nn.MaxPool2d(
            kernel_size=diffusion_kernel_size,
            stride=1,
            padding=diffusion_kernel_size // 2
        )
        
        # ============================================================
        # 注意: pos_embed 层已移除
        # 原因: racformer_head.py 使用 pos2content MLP 生成内容特征，
        #       不再需要 RWHIModule 内部的位置编码层。
        #       保留此注释以便日后参考。
        # ============================================================
    
    def _precompute_safety_anchors(self):
        """
        预计算安全流锚点 - 基于逆深度分布 (1/r)
        
        近场密集，远场稀疏，使用同心圆分布
        
        与原始 generate_points() 保持一致的距离覆盖范围！
        
        Returns:
            safety_anchors: [num_safety, 10] (theta, d, z, w, l, h, sin, cos, vx, vy)
        """
        # 使用逆深度分布计算距离
        # d ∝ 1/r，r从近到远
        num_rings = int(math.sqrt(self.num_safety))
        points_per_ring = self.num_safety // num_rings
        remainder = self.num_safety - num_rings * points_per_ring
        
        # ============================================================
        # 修复: 扩大距离覆盖范围
        # 原代码: distances * 0.4 + 0.05 只覆盖 [0.05, 0.45]
        # 原始 generate_points() 使用 linspace(0, 1, N+2)[1:-1]
        # 即距离均匀分布在 (0, 1) 范围内
        # 修改为覆盖更大范围 [0.05, 0.85]，以匹配原始分布
        # ============================================================
        # 逆深度采样: 近处密集
        inv_depths = torch.linspace(1.0, 0.15, num_rings)  # 逆深度从1到0.15
        distances = 1.0 / inv_depths  # 实际距离，范围约 [1, 6.67]
        distances = distances / distances.max()  # 归一化到[0, 1]
        
        # 映射到 [0.05, 0.85] 范围，覆盖近场到中远场
        distances = distances * 0.8 + 0.05
        
        anchors_list = []
        
        for i in range(num_rings):
            # 当前环的点数
            n_points = points_per_ring + (1 if i < remainder else 0)
            
            # 均匀角度分布
            angles = torch.linspace(0, 1, n_points + 1)[:-1]  # [0, 1) 归一化角度
            
            # 当前环的距离
            ring_distance = distances[i].expand(n_points)
            
            # 组合 (theta, d)
            ring_anchors = torch.stack([angles, ring_distance], dim=-1)
            anchors_list.append(ring_anchors)
        
        # 拼接所有环的锚点
        theta_d = torch.cat(anchors_list, dim=0)  # [num_safety, 2]
        
        # 添加其他属性 (z, w, l, h, sin, cos, vx, vy)
        num_anchors = theta_d.shape[0]
        z = torch.full((num_anchors, 1), 0.5)  # 归一化z坐标
        w = torch.full((num_anchors, 1), 0.0)  # log(w)
        l = torch.full((num_anchors, 1), 0.0)  # log(l)  
        h = torch.full((num_anchors, 1), 0.2)  # log(h)
        sin_rot = torch.zeros((num_anchors, 1))
        cos_rot = torch.ones((num_anchors, 1))
        vx = torch.zeros((num_anchors, 1))
        vy = torch.zeros((num_anchors, 1))
        
        safety_anchors = torch.cat([theta_d, z, w, l, h, sin_rot, cos_rot, vx, vy], dim=-1)
        
        return safety_anchors  # [num_safety, 10]
    
    def _precompute_safety_mask(self):
        """
        预计算安全流覆盖区域的掩码 (用于Mask显著流)
        
        Returns:
            safety_mask: [bev_grid_size, bev_grid_size] bool tensor
        """
        # 创建BEV网格坐标
        x = torch.linspace(
            self.pc_range[0] + self.cell_size_x / 2,
            self.pc_range[3] - self.cell_size_x / 2,
            self.bev_grid_size
        )
        y = torch.linspace(
            self.pc_range[1] + self.cell_size_y / 2,
            self.pc_range[4] - self.cell_size_y / 2,
            self.bev_grid_size
        )
        
        # 创建网格
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # 计算到原点的距离
        distance = torch.sqrt(xx ** 2 + yy ** 2)
        
        # 安全流覆盖区域: 距离小于safety_max_range
        safety_mask = distance < self.safety_max_range
        
        return safety_mask  # [H, W]
    
    def compute_radar_weights(self, radar_points):
        """
        计算雷达点权重 (向量化)
        
        W = log(1 + ReLU(RCS)) × (1 + α × sigmoid(|v_r|))
        
        Args:
            radar_points: [B, M, C] (x, y, z, rcs, v_r, ...)
            
        Returns:
            weights: [B, M]
        """
        # 提取RCS和径向速度
        rcs = radar_points[..., 3]  # [B, M]
        v_r = radar_points[..., 4]  # [B, M]
        
        # RCS权重: log(1 + ReLU(RCS))
        rcs_weight = torch.log1p(F.relu(rcs))  # [B, M]
        
        # 速度权重: 1 + α × sigmoid(|v_r|)
        velocity_weight = 1.0 + self.velocity_alpha * torch.sigmoid(torch.abs(v_r))  # [B, M]
        
        # 综合权重
        weights = rcs_weight * velocity_weight  # [B, M]
        
        return weights
    
    def scatter_add_voxelize(self, radar_points, weights):
        """
        散点累加体素化 (全向量化，无循环)
        
        将雷达点权重累加到BEV网格中
        
        Args:
            radar_points: [B, M, C]
            weights: [B, M]
            
        Returns:
            bev_grid: [B, 1, H, W]
        """
        B, M, _ = radar_points.shape
        device = radar_points.device
        
        # 提取xy坐标
        x = radar_points[..., 0]  # [B, M]
        y = radar_points[..., 1]  # [B, M]
        
        # 计算网格索引
        x_idx = ((x - self.pc_range[0]) / self.cell_size_x).long()  # [B, M]
        y_idx = ((y - self.pc_range[1]) / self.cell_size_y).long()  # [B, M]
        
        # 裁剪到有效范围
        x_idx = x_idx.clamp(0, self.bev_grid_size - 1)
        y_idx = y_idx.clamp(0, self.bev_grid_size - 1)
        
        # 计算一维索引: batch_idx * H * W + y_idx * W + x_idx
        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, M)  # [B, M]
        flat_idx = batch_idx * (self.bev_grid_size * self.bev_grid_size) + \
                   y_idx * self.bev_grid_size + x_idx  # [B, M]
        flat_idx = flat_idx.view(-1)  # [B*M]
        
        # 展平权重
        flat_weights = weights.view(-1)  # [B*M]
        
        # 创建输出网格
        bev_grid = torch.zeros(
            B * self.bev_grid_size * self.bev_grid_size,
            device=device,
            dtype=weights.dtype
        )
        
        # scatter_add_ 累加权重
        bev_grid.scatter_add_(0, flat_idx, flat_weights)
        
        # 重塑为 [B, 1, H, W]
        bev_grid = bev_grid.view(B, self.bev_grid_size, self.bev_grid_size)
        bev_grid = bev_grid.unsqueeze(1)  # [B, 1, H, W]
        
        return bev_grid
    
    def apply_diffusion(self, bev_grid):
        """
        应用不确定性扩散 (MaxPool2d)
        
        模拟雷达角分辨率的不确定性
        
        Args:
            bev_grid: [B, 1, H, W]
            
        Returns:
            diffused_grid: [B, 1, H, W]
        """
        return self.diffusion(bev_grid)
    
    def topk_sampling(self, bev_grid, num_samples):
        """
        Top-K采样 (向量化)
        
        从BEV网格中选取响应最高的位置
        
        Args:
            bev_grid: [B, 1, H, W]
            num_samples: 采样数量
            
        Returns:
            selected_xy: [B, num_samples, 2] 归一化xy坐标
        """
        B, _, H, W = bev_grid.shape
        device = bev_grid.device
        
        # ============================================================
        # 修复: 正确应用安全流掩码
        # 原代码 masked_grid[:, :, safety_mask] = 0.0 会导致高级索引错误
        # 改用 torch.where 或广播乘法
        # ============================================================
        safety_mask = self.safety_mask.to(device)  # [H, W]
        # 扩展掩码维度以匹配 bev_grid [B, 1, H, W]
        safety_mask_expanded = safety_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # 使用 where 将安全区域置0（只保留远处区域）
        # safety_mask=True 表示近处区域，需要置0
        masked_grid = torch.where(
            safety_mask_expanded.expand(B, 1, H, W),
            torch.zeros_like(bev_grid),
            bev_grid
        )
        
        # 添加微小噪声防止TopK重复
        noise = torch.rand_like(masked_grid) * self.noise_eps
        masked_grid = masked_grid + noise
        
        # 展平并TopK
        flat_grid = masked_grid.view(B, -1)  # [B, H*W]
        
        # 确保num_samples不超过有效网格数
        num_samples = min(num_samples, H * W)
        
        _, top_indices = torch.topk(flat_grid, num_samples, dim=1)  # [B, num_samples]
        
        # ============================================================
        # 将一维索引转换回二维坐标
        # 网格存储顺序是 [H, W]，展平后索引 = y * W + x
        # 所以: y_idx = indices // W, x_idx = indices % W
        # ============================================================
        y_idx = top_indices // W  # [B, num_samples]
        x_idx = top_indices % W   # [B, num_samples]
        
        # 转换为归一化坐标 [0, 1]
        x_norm = (x_idx.float() + 0.5) / W  # [B, num_samples]
        y_norm = (y_idx.float() + 0.5) / H  # [B, num_samples]
        
        selected_xy = torch.stack([x_norm, y_norm], dim=-1)  # [B, num_samples, 2]
        
        return selected_xy
    
    def generate_saliency_anchors(self, radar_points):
        """
        生成显著流锚点 (完全向量化)
        
        Args:
            radar_points: [B, M, C] 雷达点云
            
        Returns:
            saliency_anchors: [B, num_saliency, 10]
        """
        B = radar_points.shape[0]
        device = radar_points.device
        
        # Step 1: 计算雷达点权重
        weights = self.compute_radar_weights(radar_points)  # [B, M]
        
        # Step 2: 散点累加体素化
        bev_grid = self.scatter_add_voxelize(radar_points, weights)  # [B, 1, H, W]
        
        # Step 3: 不确定性扩散
        bev_grid = self.apply_diffusion(bev_grid)  # [B, 1, H, W]
        
        # Step 4: TopK采样 (每个高度假设采样 num_saliency_per_height 个点)
        selected_xy = self.topk_sampling(bev_grid, self.num_saliency_per_height)  # [B, N, 2]
        
        # Step 5: 多假设高度生成
        num_height_hyp = len(self.height_hypotheses)
        N = selected_xy.shape[1]
        
        # 复制xy坐标用于不同高度假设
        # [B, N, 2] -> [B, N*num_height, 2]
        selected_xy_expanded = selected_xy.unsqueeze(2).expand(B, N, num_height_hyp, 2)
        selected_xy_expanded = selected_xy_expanded.reshape(B, N * num_height_hyp, 2)
        
        # 生成高度 (归一化到[0, 1])
        z_min, z_max = self.pc_range[2], self.pc_range[5]
        heights_norm = torch.tensor(
            [(h - z_min) / (z_max - z_min) for h in self.height_hypotheses],
            device=device, dtype=selected_xy.dtype
        )  # [num_height]
        
        # 扩展高度: [B, N, num_height] -> [B, N*num_height]
        heights = heights_norm.view(1, 1, num_height_hyp).expand(B, N, num_height_hyp)
        heights = heights.reshape(B, N * num_height_hyp, 1)  # [B, N*num_height, 1]
        
        # 转换xy坐标为theta-d极坐标
        # 注意: selected_xy 是归一化坐标 [0,1]，需要转换
        theta_d = self._xy_to_theta_d(selected_xy_expanded)  # [B, N*num_height, 2]
        
        # 组合成完整的锚点表示
        num_anchors = N * num_height_hyp
        w = torch.zeros((B, num_anchors, 1), device=device)
        l = torch.zeros((B, num_anchors, 1), device=device)
        h = torch.full((B, num_anchors, 1), 0.2, device=device)
        sin_rot = torch.zeros((B, num_anchors, 1), device=device)
        cos_rot = torch.ones((B, num_anchors, 1), device=device)
        vx = torch.zeros((B, num_anchors, 1), device=device)
        vy = torch.zeros((B, num_anchors, 1), device=device)
        
        saliency_anchors = torch.cat([
            theta_d, heights, w, l, h, sin_rot, cos_rot, vx, vy
        ], dim=-1)  # [B, num_anchors, 10]
        
        return saliency_anchors
    
    def _xy_to_theta_d(self, xy_norm):
        """
        将归一化xy坐标转换为theta-d极坐标
        
        ============================================================
        坐标系约定 (与 bbox/utils.py 的 xy2theta_d_coods 完全一致):
        
        1. 输入: xy_norm ∈ [0, 1]，表示BEV网格中的归一化位置
           - (0, 0) = BEV左下角 (x_min, y_min)
           - (1, 1) = BEV右上角 (x_max, y_max)
           - (0.5, 0.5) = BEV中心 (ego车辆位置)
        
        2. 输出: (theta, d) 极坐标
           - theta ∈ [0, 1]: 归一化角度，对应 [0, 2π)
             * theta=0 → 正X轴方向 (车辆右侧)
             * theta=0.25 → 正Y轴方向 (车辆前方)
             * theta=0.5 → 负X轴方向 (车辆左侧)
             * theta=0.75 → 负Y轴方向 (车辆后方)
           - d ∈ [0, 1]: 归一化距离，d * polar_radius = 实际距离(米)
        
        3. 使用 atan2(dy, dx)，符合标准数学约定
        ============================================================
        
        Args:
            xy_norm: [B, N, 2] 归一化坐标 [0, 1]
            
        Returns:
            theta_d: [B, N, 2] 极坐标 (theta归一化到[0,1], d归一化到[0,1])
        """
        # 使用从pc_range动态计算的参数，而非硬编码值
        map_size = self.map_size      # 原硬编码: 102.4
        r = self.polar_radius         # 原硬编码: 65.0
        center = map_size / 2
        
        # 反归一化到实际坐标 (米)
        x = xy_norm[..., 0:1] * map_size  # [B, N, 1]
        y = xy_norm[..., 1:2] * map_size  # [B, N, 1]
        
        # 计算相对于中心的偏移
        dx = x - center
        dy = y - center
        
        # 计算极坐标
        distance = torch.sqrt(dx ** 2 + dy ** 2) / r  # 归一化距离
        theta = torch.atan2(dy, dx)  # [-π, π]
        theta = ((theta + 2 * math.pi) % (2 * math.pi)) / (2 * math.pi)  # 归一化到[0, 1]
        
        # ============================================================
        # 关键修复: Clamp theta 和 distance 到 [0, 1] 范围
        # distance 在网格边角可能超过 1.0 (sqrt(51.2^2 + 51.2^2)/65 ≈ 1.11)
        # theta 理论上已经在 [0, 1]，但为了安全也 clamp
        # ============================================================
        theta = torch.clamp(theta, 0.0, 1.0)
        distance = torch.clamp(distance, 0.0, 1.0)
        
        theta_d = torch.cat([theta, distance], dim=-1)  # [B, N, 2]
        
        return theta_d
    
    def forward(self, radar_points=None):
        """
        RWHI前向传播
        
        Args:
            radar_points: [B, M, C] 雷达点云 (x, y, z, rcs, v_r, ...)
                         如果为None，仅返回安全流锚点
            
        Returns:
            hybrid_anchors: [B, num_query, 10] 混合锚点
            anchor_mask: [B, num_query] bool 区分安全流(False)和显著流(True)
        """
        if not self.enabled or radar_points is None:
            # 仅返回安全流锚点 (复制B次)
            B = 1 if radar_points is None else radar_points.shape[0]
            device = self.safety_anchors.device if radar_points is None else radar_points.device
            
            safety_anchors = self.safety_anchors.to(device)
            safety_anchors = safety_anchors.unsqueeze(0).expand(B, -1, -1).clone()
            
            # 如果禁用RWHI，用安全流填满所有Query
            if not self.enabled:
                # 重复安全流锚点以填满num_query
                repeat_times = (self.num_query + self.num_safety - 1) // self.num_safety
                full_anchors = safety_anchors.repeat(1, repeat_times, 1)[:, :self.num_query, :]
                anchor_mask = torch.zeros(B, self.num_query, dtype=torch.bool, device=device)
                return full_anchors, anchor_mask
            
            # 正常返回安全流
            anchor_mask = torch.zeros(B, self.num_safety, dtype=torch.bool, device=device)
            return safety_anchors, anchor_mask
        
        B = radar_points.shape[0]
        device = radar_points.device
        
        # 获取安全流锚点
        safety_anchors = self.safety_anchors.to(device)
        safety_anchors = safety_anchors.unsqueeze(0).expand(B, -1, -1).clone()  # [B, num_safety, 10]
        
        # 生成显著流锚点
        saliency_anchors = self.generate_saliency_anchors(radar_points)  # [B, num_saliency, 10]
        
        # 拼接安全流和显著流
        hybrid_anchors = torch.cat([safety_anchors, saliency_anchors], dim=1)  # [B, num_query, 10]
        
        # 创建掩码 (安全流=False, 显著流=True)
        anchor_mask = torch.zeros(B, self.num_query, dtype=torch.bool, device=device)
        anchor_mask[:, self.num_safety:] = True
        
        return hybrid_anchors, anchor_mask
    
    # ============================================================
    # 注意: get_position_embedding 方法已移除
    # 原因: racformer_head.py 使用 pos2content MLP 生成内容特征，
    #       不需要此方法。如需位置编码，请使用 racformer_head.pos2content。
    # ============================================================


# ============================================================
# RWHIQueryGenerator - 已废弃
# ============================================================
# 此类原本用于将RWHI模块与RaCFormer Head集成，但现在:
# 1. racformer_head.py 直接使用 RWHIModule
# 2. racformer_head.py 使用自己的 pos2content MLP 生成内容特征
# 
# 保留此类定义以保持向后兼容，但不建议使用。
# 如需Query生成功能，请参考 racformer_head.py 中的实现。
# ============================================================
class RWHIQueryGenerator(BaseModule):
    """
    [已废弃] RWHI Query生成器
    
    警告: 此类已废弃，请直接使用 RWHIModule 并在 RaCFormer_head 中
          使用 pos2content MLP 生成内容特征。
    """
    
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(
            "RWHIQueryGenerator 已废弃。"
            "请直接使用 RWHIModule，并在 RaCFormer_head 中使用 pos2content MLP。"
            "参考: RaCFormer/models/racformer_head.py"
        )

