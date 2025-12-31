# ============================================================
# RaCFormer 配置文件 - 集成RWHI Query初始化策略
# ============================================================
# RWHI (RCS-Weighted Hybrid Anchor Initialization)
# 
# 特点:
# 1. 安全流 (~30%): 基于逆深度分布的固定锚点，覆盖近处盲区
# 2. 显著流 (~70%): 基于雷达RCS和速度的动态锚点
# 3. 全向量化实现，TensorRT兼容
#
# 使用方法：
#   训练：python train.py --config configs/racformer_with_rwhi.py
#   测试：python val.py --config configs/racformer_with_rwhi.py --weights checkpoints/xxx.pth
# ============================================================

# 继承基础配置
_base_ = './racformer_r50_nuimg_704x256_f8.py'

# ============================================================
# RWHI模块配置
# ============================================================
rwhi_cfg = dict(
    # 锚点分配
    safety_ratio=0.3,            # 安全流占比 (30%)
    
    # 空间范围
    bev_grid_size=100,           # BEV网格分辨率 (100x100)
    safety_max_range=30.0,       # 安全流最大范围 (米)
    
    # 权重计算
    velocity_alpha=0.5,          # 速度权重系数
    
    # 高度假设 (米) - 多假设提高召回率
    height_hypotheses=(0.0, 1.5),  # 地面 + 典型车辆高度
    
    # 不确定性扩散
    diffusion_kernel_size=3,     # MaxPool核大小
    
    # TopK防重复噪声
    noise_eps=1e-6,
    
    # 总开关
    enabled=True,
)

# ============================================================
# 模型配置 - 启用RWHI
# ============================================================
model = dict(
    pts_bbox_head=dict(
        # 启用RWHI Query初始化
        use_rwhi=True,
        rwhi_cfg=rwhi_cfg,
    ),
)

# ============================================================
# 调参建议
# ============================================================
#
# 【提升远距离检测】
# - 减小 safety_ratio 到 0.2 (更多显著流锚点)
# - 增大 safety_max_range 到 40.0
#
# 【提升小目标检测】
# - 增加高度假设: height_hypotheses=(0.0, 0.8, 1.5, 2.5)
# - 减小 bev_grid_size 到 80 (更粗粒度聚合)
#
# 【加速推理】
# - 减小 bev_grid_size 到 50
# - 减少 height_hypotheses 到单一值
#
# 【显存不足】
# - 减小 bev_grid_size
# - 减少 height_hypotheses
# ============================================================

