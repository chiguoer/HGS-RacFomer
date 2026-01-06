# ==============================================================================
# RWHI v2.0 模块 (Robust Hybrid Anchor Initialization)
# ==============================================================================
# 版本: 2.0
# 策略: 叠加策略 (P_total = P_base + P_radar)
# 
# 关键改进 (vs v1.0):
# - v1.0: 分割策略 (近场=安全流, 远场=雷达) -> 远场召回率崩溃
# - v2.0: 叠加策略 (基础锚点覆盖全图 + 雷达锚点增益) -> 100%空间覆盖
#
# 数学模型:
# 1. P_base(r) = max(α·r^{-1}, ε_floor)  -- 逆深度分布 + 最小密度保证
# 2. P_radar(x,y) = Σ w_i · N(x_i, y_i)  -- 高RCS区域增益
#
# 设计原则: 全向量化，无Python for循环，TensorRT兼容
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv.runner import BaseModule


class RWHIModule(BaseModule):
    """
    RWHI v2.0: Robust Hybrid Anchor Initialization Module
    
    叠加策略: P_total = P_base + P_radar
    
    锚点预算分配:
    - N_base = 500: 静态基础锚点 (安全网，覆盖0-55m全图)
    - N_radar = 400: 动态雷达锚点 (增益，聚焦高RCS区域)
    - N_total = 900
    
    基础锚点分布 (P_base):
    - 近场 (2-15m): 密集采样 (~2m stride)，约60%的点
    - 中场 (15-30m): 中等密度 (~3m stride)，约25%的点
    - 远场 (30-55m): 稀疏但连续覆盖 (~5m stride)，约15%的点
    
    雷达增益 (P_radar):
    - 仅对高RCS点 (RCS > threshold) 生效
    - 使用scatter_add体素化 + MaxPool扩散 + TopK采样
    
    输入:
        - radar_points: 雷达点云 [B, M, C] (x, y, z, rcs, v_r, ...)
        
    输出:
        - query_positions: 混合锚点位置 [B, N_total, 10]
        - anchor_mask: 锚点类型掩码 [B, N_total] (False=基础, True=雷达)
    """
    
    def __init__(self,
                 num_query=900,
                 num_base=500,           # 基础锚点数量 (安全网)
                 num_radar=400,          # 雷达锚点数量 (增益)
                 embed_dims=256,
                 pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                 bev_grid_size=100,      # 用于雷达增益的BEV网格大小
                 max_range=55.0,         # 最大感知范围 (米)
                 min_range=2.0,          # 最小感知范围 (米)
                 rcs_threshold=0.0,      # RCS过滤阈值
                 velocity_alpha=0.5,     # 速度权重系数
                 height_hypotheses=(0.0, 1.0),  # 高度假设 (米)
                 diffusion_kernel_size=3,
                 noise_eps=1e-6,
                 epsilon_floor=0.02,     # 远场最小密度参数
                 enabled=True,
                 # 兼容v1.0的参数 (忽略但不报错)
                 safety_ratio=None,
                 safety_max_range=None,
                 init_cfg=None):
        """
        初始化 RWHI v2.0 模块
        
        Args:
            num_query: 总Query数量 (默认900)
            num_base: 基础锚点数量 (默认500，静态，覆盖全图)
            num_radar: 雷达锚点数量 (默认400，动态，基于雷达增益)
            embed_dims: 嵌入维度
            pc_range: 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
            bev_grid_size: BEV网格大小
            max_range: 最大感知范围 (米)，基础锚点覆盖到此距离
            min_range: 最小感知范围 (米)，基础锚点从此距离开始
            rcs_threshold: RCS过滤阈值 (仅RCS > threshold的点参与增益)
            velocity_alpha: 速度权重系数
            height_hypotheses: 高度假设 (米)
            diffusion_kernel_size: 不确定性扩散核大小
            noise_eps: TopK防重复噪声
            epsilon_floor: 远场最小密度参数 (未直接使用，通过分段采样实现)
            enabled: 是否启用RWHI
        """
        super(RWHIModule, self).__init__(init_cfg=init_cfg)
        
        self.num_query = num_query
        self.num_base = num_base
        self.num_radar = num_radar
        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.bev_grid_size = bev_grid_size
        self.max_range = max_range
        self.min_range = min_range
        self.rcs_threshold = rcs_threshold
        self.velocity_alpha = velocity_alpha
        self.height_hypotheses = height_hypotheses
        self.diffusion_kernel_size = diffusion_kernel_size
        self.noise_eps = noise_eps
        self.epsilon_floor = epsilon_floor
        self.enabled = enabled
        
        # 验证锚点数量配置
        if num_base + num_radar != num_query:
            print(f"[RWHI v2.0] Warning: num_base({num_base}) + num_radar({num_radar}) "
                  f"!= num_query({num_query}). Adjusting num_radar.")
            self.num_radar = num_query - num_base
        
        # ============================================================
        # 计算BEV网格参数
        # ============================================================
        self.bev_x_range = pc_range[3] - pc_range[0]  # 102.4
        self.bev_y_range = pc_range[4] - pc_range[1]  # 102.4
        self.cell_size_x = self.bev_x_range / bev_grid_size
        self.cell_size_y = self.bev_y_range / bev_grid_size
        
        # 极坐标转换参数
        self.map_size = self.bev_x_range
        self.polar_radius = 65.0  # 与原始RaCFormer保持一致
        
        # ============================================================
        # 预计算基础锚点 (覆盖全图的安全网)
        # 关键改进: 范围 [min_range, max_range] 而非 [0, 30m]
        # ============================================================
        base_anchors = self._precompute_base_anchors()
        self.register_buffer('base_anchors', base_anchors)
        
        # 不确定性扩散层 (MaxPool2d)
        self.diffusion = nn.MaxPool2d(
            kernel_size=diffusion_kernel_size,
            stride=1,
            padding=diffusion_kernel_size // 2
        )
        
        print(f"[RWHI v2.0] Initialized: num_base={self.num_base}, num_radar={self.num_radar}, "
              f"range=[{min_range}, {max_range}]m")
    
    def _precompute_base_anchors(self):
        """
        预计算基础锚点 - 遵循逆深度分布 P_base(r) = max(α·r^{-1}, ε_floor)
        
        关键设计:
        1. 覆盖范围: [min_range, max_range] (e.g., 2m - 55m)
        2. 密度分布: 近场密集 (~2m stride), 远场稀疏但连续 (~5m stride)
        3. 使用极坐标采样: 同心环 × 角度射线
        
        分段采样策略:
        - 近场 (2-15m): 60% 的环，~2m间距
        - 中场 (15-30m): 25% 的环，~3m间距
        - 远场 (30-55m): 15% 的环，~5m间距 (关键改进!)
        
        Returns:
            base_anchors: [num_base, 10] (theta, d, z, w, l, h, sin, cos, vx, vy)
        """
        # ============================================================
        # Step 1: 计算环数和角度数
        # ============================================================
        # 使用黄金比例分配以平衡角度和距离采样
        aspect_ratio = 1.2  # 角度数 / 环数 的比例
        num_rings = int(math.sqrt(self.num_base / aspect_ratio))
        num_angles = self.num_base // num_rings
        
        # 确保总点数足够
        while num_rings * num_angles < self.num_base:
            num_angles += 1
        
        # ============================================================
        # Step 2: 分段距离采样 (逆深度分布近似)
        # ============================================================
        # 近场 (min_range - 15m): 60% 的环
        # 中场 (15m - 30m): 25% 的环
        # 远场 (30m - max_range): 15% 的环 (关键: 覆盖到55m!)
        
        near_ratio = 0.55
        mid_ratio = 0.30
        far_ratio = 0.15
        
        near_rings = max(int(num_rings * near_ratio), 3)
        mid_rings = max(int(num_rings * mid_ratio), 2)
        far_rings = max(int(num_rings * far_ratio), 3)  # 确保远场至少3环
        
        # 调整以匹配总环数
        total_rings = near_rings + mid_rings + far_rings
        if total_rings > num_rings:
            # 按比例缩减
            scale = num_rings / total_rings
            near_rings = max(int(near_rings * scale), 2)
            mid_rings = max(int(mid_rings * scale), 2)
            far_rings = num_rings - near_rings - mid_rings
        elif total_rings < num_rings:
            # 增加近场
            near_rings += num_rings - total_rings
        
        r = self.polar_radius  # 65.0m
        
        # 生成各区域的归一化距离 [0, 1]
        # 近场: min_range - 15m
        near_start = self.min_range / r
        near_end = 15.0 / r
        near_distances = torch.linspace(near_start, near_end, near_rings + 1)[:-1]
        
        # 中场: 15m - 30m
        mid_start = 15.0 / r
        mid_end = 30.0 / r
        mid_distances = torch.linspace(mid_start, mid_end, mid_rings + 1)[:-1]
        
        # 远场: 30m - max_range (关键改进: 覆盖到55m!)
        far_start = 30.0 / r
        far_end = min(self.max_range, 55.0) / r
        far_distances = torch.linspace(far_start, far_end, far_rings + 1)[:-1]
        
        all_distances = torch.cat([near_distances, mid_distances, far_distances])
        actual_num_rings = all_distances.shape[0]
        
        # ============================================================
        # Step 3: 生成角度 (均匀分布在 [0, 2π))
        # ============================================================
        angles = torch.linspace(0, 1, num_angles + 1)[:-1]  # [0, 1)
        
        # ============================================================
        # Step 4: 创建网格 (向量化广播)
        # ============================================================
        # distances: [actual_num_rings]
        # angles: [num_angles]
        # -> theta_grid: [actual_num_rings, num_angles]
        # -> d_grid: [actual_num_rings, num_angles]
        
        theta_grid = angles.view(1, num_angles).expand(actual_num_rings, num_angles)
        d_grid = all_distances.view(actual_num_rings, 1).expand(actual_num_rings, num_angles)
        
        # 展平
        theta = theta_grid.reshape(-1)
        d = d_grid.reshape(-1)
        
        # ============================================================
        # Step 5: 调整到 num_base
        # ============================================================
        current_num = theta.shape[0]
        
        if current_num >= self.num_base:
            # 截断
            theta = theta[:self.num_base]
            d = d[:self.num_base]
        else:
            # 填充: 在远场区域添加更多点
            extra_needed = self.num_base - current_num
            # 使用不同角度偏移在远场添加点
            extra_angles = torch.linspace(0.5 / num_angles, 1 - 0.5 / num_angles, extra_needed)
            extra_d = torch.full((extra_needed,), far_end * 0.9)  # 远场距离
            
            theta = torch.cat([theta, extra_angles])
            d = torch.cat([d, extra_d])
        
        # ============================================================
        # Step 6: 组合锚点属性
        # ============================================================
        theta_d = torch.stack([theta, d], dim=-1)  # [num_base, 2]
        
        num_anchors = theta_d.shape[0]
        z = torch.full((num_anchors, 1), 0.5)  # 归一化z坐标 (中间高度)
        w = torch.full((num_anchors, 1), 0.0)  # log(w)
        l = torch.full((num_anchors, 1), 0.0)  # log(l)  
        h = torch.full((num_anchors, 1), 0.2)  # log(h)
        sin_rot = torch.zeros((num_anchors, 1))
        cos_rot = torch.ones((num_anchors, 1))
        vx = torch.zeros((num_anchors, 1))
        vy = torch.zeros((num_anchors, 1))
        
        base_anchors = torch.cat([theta_d, z, w, l, h, sin_rot, cos_rot, vx, vy], dim=-1)
        
        # 打印分布统计
        print(f"[RWHI v2.0] Base anchors: {num_anchors} points, "
              f"rings={actual_num_rings} (near={near_rings}, mid={mid_rings}, far={far_rings}), "
              f"angles={num_angles}")
        print(f"[RWHI v2.0] Distance range: [{d.min().item()*r:.1f}m, {d.max().item()*r:.1f}m]")
        
        return base_anchors  # [num_base, 10]
    
    # ============================================================
    # 雷达权重计算
    # ============================================================
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
    
    def filter_by_rcs(self, radar_points, weights):
        """
        根据RCS阈值过滤雷达点 (向量化)
        
        仅保留高置信度点 (RCS > threshold) 的权重
        
        Args:
            radar_points: [B, M, C]
            weights: [B, M]
            
        Returns:
            filtered_weights: [B, M] (低RCS点的权重置为0)
        """
        rcs = radar_points[..., 3]  # [B, M]
        
        # 创建掩码: RCS > threshold
        valid_mask = (rcs > self.rcs_threshold).float()  # [B, M]
        
        # 应用掩码
        filtered_weights = weights * valid_mask  # [B, M]
        
        return filtered_weights
    
    # ============================================================
    # 体素化和采样
    # ============================================================
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
        
        从BEV网格中选取响应最高的位置作为雷达增益锚点
        
        Args:
            bev_grid: [B, 1, H, W]
            num_samples: 采样数量
            
        Returns:
            selected_xy: [B, num_samples, 2] 归一化xy坐标 [0, 1]
        """
        B, _, H, W = bev_grid.shape
        device = bev_grid.device
        
        # 添加微小噪声防止TopK重复
        noise = torch.rand_like(bev_grid) * self.noise_eps
        noisy_grid = bev_grid + noise
        
        # 展平并TopK
        flat_grid = noisy_grid.view(B, -1)  # [B, H*W]
        
        # 确保num_samples不超过有效网格数
        num_samples = min(num_samples, H * W)
        
        _, top_indices = torch.topk(flat_grid, num_samples, dim=1)  # [B, num_samples]
        
        # 将一维索引转换回二维坐标
        y_idx = top_indices // W  # [B, num_samples]
        x_idx = top_indices % W   # [B, num_samples]
        
        # 转换为归一化坐标 [0, 1]
        x_norm = (x_idx.float() + 0.5) / W  # [B, num_samples]
        y_norm = (y_idx.float() + 0.5) / H  # [B, num_samples]
        
        selected_xy = torch.stack([x_norm, y_norm], dim=-1)  # [B, num_samples, 2]
        
        return selected_xy
    
    # ============================================================
    # 坐标转换
    # ============================================================
    def _xy_to_theta_d(self, xy_norm):
        """
        将归一化xy坐标转换为theta-d极坐标
        
        坐标系约定:
        - xy_norm ∈ [0, 1]，BEV网格归一化位置
        - (0.5, 0.5) = ego车辆位置 (BEV中心)
        - theta ∈ [0, 1]: 归一化角度 [0, 2π)
        - d ∈ [0, 1]: 归一化距离
        
        Args:
            xy_norm: [B, N, 2] 归一化坐标 [0, 1]
            
        Returns:
            theta_d: [B, N, 2] 极坐标
        """
        map_size = self.map_size
        r = self.polar_radius
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
        
        # Clamp 到 [0, 1] 范围
        theta = torch.clamp(theta, 0.0, 1.0)
        distance = torch.clamp(distance, 0.0, 1.0)
        
        theta_d = torch.cat([theta, distance], dim=-1)  # [B, N, 2]
        
        return theta_d
    
    # ============================================================
    # 锚点获取方法
    # ============================================================
    def get_base_anchors(self, batch_size, device):
        """
        获取基础锚点 (静态，覆盖全图 0-55m)
        
        Args:
            batch_size: batch大小
            device: 设备
            
        Returns:
            base_anchors: [B, num_base, 10]
        """
        base_anchors = self.base_anchors.to(device)
        base_anchors = base_anchors.unsqueeze(0).expand(batch_size, -1, -1).clone()
        return base_anchors
    
    def get_radar_anchors(self, radar_points):
        """
        获取雷达增益锚点 (动态，基于高RCS区域)
        
        流程:
        1. 计算雷达点权重 W = log(1+RCS) × (1 + α×sigmoid(|v_r|))
        2. 按RCS阈值过滤
        3. scatter_add 体素化到BEV网格
        4. MaxPool 不确定性扩散
        5. TopK 采样高价值区域
        6. 多高度假设生成
        
        Args:
            radar_points: [B, M, C] 雷达点云
            
        Returns:
            radar_anchors: [B, num_radar, 10]
        """
        B = radar_points.shape[0]
        device = radar_points.device
        
        # Step 1: 计算雷达点权重
        weights = self.compute_radar_weights(radar_points)  # [B, M]
        
        # Step 2: 根据RCS阈值过滤
        weights = self.filter_by_rcs(radar_points, weights)  # [B, M]
        
        # Step 3: 散点累加体素化
        bev_grid = self.scatter_add_voxelize(radar_points, weights)  # [B, 1, H, W]
        
        # Step 4: 不确定性扩散
        bev_grid = self.apply_diffusion(bev_grid)  # [B, 1, H, W]
        
        # Step 5: TopK采样
        # 考虑高度假设，每个假设采样 num_radar / num_heights 个点
        num_heights = len(self.height_hypotheses)
        num_samples_per_height = self.num_radar // num_heights
        
        selected_xy = self.topk_sampling(bev_grid, num_samples_per_height)  # [B, N, 2]
        
        # Step 6: 多假设高度生成
        N = selected_xy.shape[1]
        
        # 复制xy坐标用于不同高度假设 (向量化)
        # [B, N, 2] -> [B, N, num_heights, 2] -> [B, N*num_heights, 2]
        selected_xy_expanded = selected_xy.unsqueeze(2).expand(B, N, num_heights, 2)
        selected_xy_expanded = selected_xy_expanded.reshape(B, N * num_heights, 2)
        
        # 生成高度 (归一化到[0, 1])
        z_min, z_max = self.pc_range[2], self.pc_range[5]
        heights_norm = torch.tensor(
            [(h - z_min) / (z_max - z_min) for h in self.height_hypotheses],
            device=device, dtype=selected_xy.dtype
        )  # [num_heights]
        
        # 扩展高度
        # [num_heights] -> [1, 1, num_heights] -> [B, N, num_heights] -> [B, N*num_heights, 1]
        heights = heights_norm.view(1, 1, num_heights).expand(B, N, num_heights)
        heights = heights.reshape(B, N * num_heights, 1)
        
        # 转换xy坐标为theta-d极坐标
        theta_d = self._xy_to_theta_d(selected_xy_expanded)  # [B, N*num_heights, 2]
        
        # 组合成完整的锚点表示
        num_anchors = N * num_heights
        w = torch.zeros((B, num_anchors, 1), device=device)
        l = torch.zeros((B, num_anchors, 1), device=device)
        h = torch.full((B, num_anchors, 1), 0.2, device=device)
        sin_rot = torch.zeros((B, num_anchors, 1), device=device)
        cos_rot = torch.ones((B, num_anchors, 1), device=device)
        vx = torch.zeros((B, num_anchors, 1), device=device)
        vy = torch.zeros((B, num_anchors, 1), device=device)
        
        radar_anchors = torch.cat([
            theta_d, heights, w, l, h, sin_rot, cos_rot, vx, vy
        ], dim=-1)  # [B, num_anchors, 10]
        
        # 确保锚点数量正确 (截断或填充)
        if radar_anchors.shape[1] > self.num_radar:
            radar_anchors = radar_anchors[:, :self.num_radar, :]
        elif radar_anchors.shape[1] < self.num_radar:
            # 用默认值填充
            padding_size = self.num_radar - radar_anchors.shape[1]
            padding = torch.zeros(B, padding_size, 10, device=device, dtype=radar_anchors.dtype)
            padding[..., 2] = 0.5  # z
            padding[..., 7] = 1.0  # cos
            radar_anchors = torch.cat([radar_anchors, padding], dim=1)
        
        return radar_anchors
    
    # ============================================================
    # 前向传播
    # ============================================================
    def forward(self, radar_points=None):
        """
        RWHI v2.0 前向传播
        
        叠加策略: query_bbox = concat(base_anchors, radar_anchors)
        - base_anchors: 静态安全网，覆盖全图 (0-55m)
        - radar_anchors: 动态增益，聚焦高RCS区域
        
        Args:
            radar_points: [B, M, C] 雷达点云 (x, y, z, rcs, v_r, ...)
                         如果为None，仅返回基础锚点 (复制以填满num_query)
            
        Returns:
            hybrid_anchors: [B, num_query, 10] 混合锚点
            anchor_mask: [B, num_query] bool 区分基础(False)和雷达(True)
        """
        # ============================================================
        # Case 1: RWHI禁用
        # ============================================================
        if not self.enabled:
            B = 1 if radar_points is None else radar_points.shape[0]
            device = self.base_anchors.device if radar_points is None else radar_points.device
            
            base_anchors = self.base_anchors.to(device)
            base_anchors = base_anchors.unsqueeze(0).expand(B, -1, -1).clone()
            
            # 重复基础锚点以填满num_query
            repeat_times = (self.num_query + self.num_base - 1) // self.num_base
            full_anchors = base_anchors.repeat(1, repeat_times, 1)[:, :self.num_query, :]
            anchor_mask = torch.zeros(B, self.num_query, dtype=torch.bool, device=device)
            return full_anchors, anchor_mask
        
        # ============================================================
        # Case 2: 无雷达数据
        # ============================================================
        if radar_points is None:
            B = 1
            device = self.base_anchors.device
            
            base_anchors = self.base_anchors.to(device)
            base_anchors = base_anchors.unsqueeze(0).expand(B, -1, -1).clone()
            
            # 重复基础锚点以填满num_query
            repeat_times = (self.num_query + self.num_base - 1) // self.num_base
            full_anchors = base_anchors.repeat(1, repeat_times, 1)[:, :self.num_query, :]
            anchor_mask = torch.zeros(B, self.num_query, dtype=torch.bool, device=device)
            return full_anchors, anchor_mask
        
        # ============================================================
        # Case 3: 正常叠加策略
        # ============================================================
        B = radar_points.shape[0]
        device = radar_points.device
        
        # 获取基础锚点 (静态，覆盖全图)
        base_anchors = self.get_base_anchors(B, device)  # [B, num_base, 10]
        
        # 获取雷达增益锚点 (动态，基于高RCS区域)
        radar_anchors = self.get_radar_anchors(radar_points)  # [B, num_radar, 10]
        
        # 拼接: 叠加策略 P_total = P_base + P_radar
        hybrid_anchors = torch.cat([base_anchors, radar_anchors], dim=1)  # [B, num_query, 10]
        
        # 创建掩码 (基础=False, 雷达=True)
        anchor_mask = torch.zeros(B, self.num_query, dtype=torch.bool, device=device)
        anchor_mask[:, self.num_base:] = True
        
        return hybrid_anchors, anchor_mask
    
    # ============================================================
    # 兼容性属性 (供 racformer_head.py 使用)
    # ============================================================
    @property
    def safety_anchors(self):
        """
        兼容性属性: 返回基础锚点作为"安全锚点"
        
        供 racformer_head.py 初始化 init_query_bbox 使用
        """
        return self.base_anchors
    
    @property
    def num_safety(self):
        """
        兼容性属性: 返回基础锚点数量
        
        供 racformer_head.py 使用
        """
        return self.num_base


# ============================================================
# RWHIQueryGenerator - 已废弃
# ============================================================
class RWHIQueryGenerator(BaseModule):
    """
    [已废弃] RWHI Query生成器
    
    警告: 此类已废弃，请直接使用 RWHIModule。
    保留此类定义以保持向后兼容，但不建议使用。
    """
    
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(
            "RWHIQueryGenerator 已废弃。"
            "请直接使用 RWHIModule，并在 RaCFormer_head 中使用 pos2content MLP。"
            "参考: RaCFormer/models/racformer_head.py"
        )
