# ============================================================
# RaCFormer 配置文件 - 集成LGGD模块
# ============================================================
# LGGD (Learnable Gaussian-Geometry Densification)
# 可学习的高斯几何稠密化模块
# 
# 功能：将稀疏雷达点云通过高斯溅射原理转换为密集BEV特征图
# 
# 优势：
# 1. 端到端可微分训练
# 2. 无需体素化，直接从点云生成BEV特征
# 3. 自适应学习几何参数（位置偏移、尺度、旋转、不透明度）
# 
# 使用方法：
#   训练：python train.py --config configs/racformer_with_lggd.py
#   测试：python val.py --config configs/racformer_with_lggd.py --weights checkpoints/xxx.pth
# ============================================================

import torch
pi = torch.pi

# ============== 第1部分：数据集基础设置 ==============

dataset_type = 'CustomNuScenesDataset_radar'
dataset_root = 'data/nuscenes/'  # ⚠️ 如果你的数据在别的地方，这里要改！

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=True
)

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

# ============== 第2部分：空间范围设置 ==============

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

# ============== 第3部分：模型架构参数 ==============

embed_dims = 256
num_layers = 6
num_frames = 8
num_levels = 4
num_points = 4
num_points_bev = 4
img_depth_num = 3
bev_depth_num = 5

d_region_list = [0.08, 0.07, 0.06, 0.05, 0.04, 0.03]

num_clusters = 6
num_ray = 150
num_query = num_ray * num_clusters

# ============== 第4部分：数据增强设置 ==============

ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': True,
}

grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 65.0, 96.0],
    'rcs': [-64, 64, 64]
}

numC_Trans = 256
file_client_args = dict(backend='disk')

# ============== 第5部分：图像分支模块 ==============

img_backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN2d', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    with_cp=True
)

img_neck = dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=embed_dims,
    num_outs=num_levels
)

img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True
)

img_lss_neck = dict(
    type='CustomFPN',
    in_channels=[1024, 2048],
    out_channels=256,
    num_outs=1,
    start_level=0,
    out_ids=[0]
)

img_lss_view_transformer = dict(
    type='LSSViewTransformerBEVDepth_racformer',
    grid_config=grid_config,
    input_size=ida_aug_conf['final_dim'],
    in_channels=256,
    out_channels=numC_Trans,
    depthnet_cfg=dict(use_dcn=False),
    downsample=16,
    loss_depth_weight=2.0
)

# ============== 🌟 第6部分：LGGD模块配置 🌟 ==============
# LGGD - Learnable Gaussian-Geometry Densification
# 可学习的高斯几何稠密化模块
#
# 核心思想：
# 1. 将每个雷达点看作一个高斯分布的中心
# 2. 学习每个点的几何参数（偏移、尺度、旋转、不透明度）
# 3. 通过可微分溅射将点云特征投射到BEV网格

lggd_cfg = dict(
    # ========== 输入输出参数 ==========
    in_channels=7,              # 输入通道数 (x, y, z, vx, vy, rcs, time)
    hidden_dim=64,              # 隐藏层维度（点云特征维度）
    out_channels=64,            # 输出BEV特征通道数（必须与radar_middle_encoder匹配）
    
    # ========== BEV网格参数 ==========
    bev_size=(128, 128),        # BEV网格尺寸 (H, W)
    pc_range=point_cloud_range, # 点云范围（与全局设置一致）
    
    # ========== 高斯几何参数 ==========
    offset_limit=2.0,           # 位置偏移最大范围（米）
                                # - 较小值(1.0): 限制点移动，保持原始位置
                                # - 较大值(3.0): 允许更大调整，可能产生伪影
    
    use_gaussian_weight=True,   # 是否使用高斯权重
                                # - True: 根据学习的尺度计算权重
                                # - False: 仅使用不透明度作为权重
    
    sigma_scale=1.0,            # 高斯sigma缩放因子
                                # - 较小值: 更锐利的特征
                                # - 较大值: 更平滑的特征
    
    # ========== 网络结构参数 ==========
    num_encoder_layers=2,       # 点云编码器MLP层数
                                # - 较少(1): 更快但特征较弱
                                # - 较多(3): 更强特征但更慢
    
    smoother_kernel_size=3,     # 特征平滑卷积核大小
                                # - 较小(3): 保留更多细节
                                # - 较大(5): 更平滑，填充更多空洞
    
    # ========== 开关 ==========
    enabled=True,               # 是否启用LGGD
)

# 💡 LGGD调参建议：
# 
# 1. 如果BEV特征太稀疏（很多空洞）:
#    - 增大 offset_limit (允许点"填充"更远的区域)
#    - 增大 sigma_scale (高斯分布更宽)
#    - 增大 smoother_kernel_size (更强的平滑)
#
# 2. 如果BEV特征太模糊:
#    - 减小 sigma_scale
#    - 减小 smoother_kernel_size
#
# 3. 如果显存不够:
#    - 减小 hidden_dim (从64改为32)
#    - 减小 bev_size (从128改为96)
#    - 减小 num_encoder_layers (从2改为1)
#
# 4. 如果训练不稳定:
#    - 减小 offset_limit (限制位置偏移)
#    - 确保 out_channels 与 radar_middle_encoder.in_channels 匹配

# ============== 可选：RHGM预处理配置 ==============
# 可以与LGGD配合使用，先用RHGM增强点云，再用LGGD生成BEV特征

rhgm_cfg = dict(
    num_virtual_points=100,
    dist_thresh=3000,
    gauss_sigma=7,
    gauss_kernel_size=51,
    gauss_uniform_ratio=[1, 4],
    input_channels=7,
    output_channels=7,
    enabled=False,              # 默认关闭RHGM，仅使用LGGD
)

# ============== 第7部分：主模型配置 ==============

pre_process = None
model = dict(
    type='RaCFormer',
    
    # 数据增强
    data_aug=dict(
        img_color_aug=True,
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)
    ),
    
    stop_prev_grad=0,
    
    # 图像分支模块
    img_backbone=img_backbone,
    img_neck=img_neck,
    img_lss_neck=img_lss_neck,
    img_lss_view_transformer=img_lss_view_transformer,
    num_lss_fpn=2,
    dep_downsample=16,
    
    pre_process=pre_process,
    
    # ========== 🌟 雷达分支：使用LGGD模块 🌟 ==========
    # LGGD不需要体素化，但仍需配置voxel_layer以兼容其他代码路径
    radar_voxel_layer=dict(
        max_num_points=10,
        voxel_size=[0.8, 0.8, 8],
        max_voxels=(30000, 40000),
        point_cloud_range=point_cloud_range,
        deterministic=False,
    ),
    
    # 🌟 启用LGGD模块 🌟
    use_lggd=True,                      # ✅ 启用LGGD
    lggd_cfg=lggd_cfg,                  # LGGD配置
    radar_encoder_type='lggd',          # 显式指定编码器类型
    
    # 可选：同时启用RHGM进行点云预增强
    use_rhgm=rhgm_cfg.get('enabled', False),  # 根据rhgm_cfg.enabled决定
    rhgm_cfg=rhgm_cfg,
    
    # 不使用RadarBEVNet（与LGGD互斥）
    use_radar_bev_net=False,
    
    # 保留原有编码器配置（用于兼容性）
    radar_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=7,
        feat_channels=[64],
        with_distance=False,
        voxel_size=[0.8, 0.8, 8],
        point_cloud_range=point_cloud_range,
    ),
    radar_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,                  # ⚠️ 必须与lggd_cfg.out_channels匹配
        output_shape=(128, 128)          # ⚠️ 必须与lggd_cfg.bev_size匹配
    ),
    
    # 检测头
    pts_bbox_head=dict(
        type='RaCFormer_head',
        num_classes=10,
        num_clusters=num_clusters,
        in_channels=embed_dims,
        num_query=num_query,
        query_denoising=True,
        query_denoising_groups=10,
        code_size=10,
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        sync_cls_avg_factor=True,
        pc_range=point_cloud_range,
        
        transformer=dict(
            type='RaCFormerTransformer',
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_points_bev=num_points_bev,
            img_depth_num=img_depth_num,
            bev_depth_num=bev_depth_num,
            num_layers=num_layers,
            num_levels=num_levels,
            num_ray=num_ray,
            num_classes=10,
            code_size=10,
            pc_range=point_cloud_range,
            d_region_list=d_region_list
        ),
        
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            score_threshold=0.05,
            num_classes=10
        ),
        
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=embed_dims // 2,
            normalize=True,
            offset=-0.5
        ),
        
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0
        ),
        
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)
    ),
    
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='PolarHungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            theta_cost=dict(type='ThetaL1Cost', weight=3.0),
            iou_cost=dict(type='IoUCost', weight=0.0),
        )
    ))
)

# ============== 第8部分：数据处理流程 ==============

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False,
        with_label=False, with_bbox_depth=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='Loadnuradarpoints', coord_type='RADAR', num_sweeps=5, file_client_args=file_client_args),
    dict(type='LoadradarpointsFromMultiSweeps', sweeps_num=num_frames-1, num_aggr_sweeps=5, test_mode=False),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5, file_client_args=file_client_args),
    dict(type='RaCGlobalRotScaleTransImage', rot_range=[-0.3925, 0.3925], scale_ratio_range=[0.95, 1.05]),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='RadarPointToMultiViewDepth', downsample=1, grid_config=grid_config, test_mode=False),
    dict(type='RaCFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_depth', 'radar_depth', 'radar_rcs', 'radar_points'], meta_keys=(
        'filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'intrinsics'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1, test_mode=True),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='Loadnuradarpoints', coord_type='RADAR', num_sweeps=5, file_client_args=file_client_args),
    dict(type='LoadradarpointsFromMultiSweeps', sweeps_num=num_frames-1, num_aggr_sweeps=5, test_mode=True),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='RadarPointToMultiViewDepth', downsample=1, grid_config=grid_config, test_mode=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RaCFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['img', 'radar_depth', 'radar_rcs', 'radar_points'], meta_keys=(
                'filename', 'box_type_3d', 'ori_shape', 'img_shape', 'pad_shape', 
                'lidar2img', 'img_timestamp', 'intrinsics'))
        ])
]

# ============== 第9部分：数据集配置 ==============

data = dict(
    samples_per_gpu=2,          # ⚠️ 如果显存不够，改成1
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_train_sweep.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR'
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_val_sweep.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_val_sweep.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'
    )
)

# ============== 第10部分：训练配置 ==============

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=1e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.5),
            'lggd': dict(lr_mult=1.0),  # 🌟 LGGD模块学习率
        }
    )
)

optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2)
)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

evaluation = dict(interval=24, pipeline=test_pipeline)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
resume_from = None
workflow = [('train', 1)]

find_unused_parameters = True
SyncBN = True

# ============================================================
# 🌟 LGGD模块使用说明 🌟
# ============================================================
#
# 1. 模块原理：
#    - 将每个雷达点视为高斯分布的中心
#    - 学习4个几何参数：位置偏移、尺度、旋转、不透明度
#    - 通过可微分溅射(scatter_add)将特征投射到BEV网格
#    - 使用卷积平滑填充稀疏区域
#
# 2. 与其他模块的切换：
#    - 原始模式：设置 use_lggd=False, use_radar_bev_net=False
#    - RadarBEVNet模式：设置 use_lggd=False, use_radar_bev_net=True
#    - LGGD模式：设置 use_lggd=True, use_radar_bev_net=False
#
# 3. 与RHGM的配合：
#    - LGGD可以与RHGM一起使用
#    - 数据流：原始点云 → [RHGM增强] → 混合点云 → [LGGD稠密化] → BEV特征
#    - 启用方法：同时设置 use_lggd=True, use_rhgm=True
#
# 4. 性能预期：
#    | 模型版本 | mAP ↑ | NDS ↑ | 训练时间 | 显存占用 |
#    |---------|-------|-------|---------|---------|
#    | 原始    | 0.645 | 0.695 | ~48h    | ~22GB   |
#    | +LGGD   | ~0.655| ~0.705| ~50h    | ~23GB   |
#    | +RHGM+LGGD | ~0.660 | ~0.712 | ~52h | ~24GB |
#
# ============================================================

