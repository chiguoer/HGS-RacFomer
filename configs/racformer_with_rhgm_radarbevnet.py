# ============================================================
# RaCFormer 配置文件 - 集成RHGM和RadarBEVNet模块
# ============================================================
# 这个配置文件就像"菜谱"，告诉程序怎么"烹饪"一个完整的检测模型
# 
# 新增功能：
# 1. RHGM模块：让雷达点云更"丰富"（加虚拟点）
# 2. RadarBEVNet模块：用更强大的方法处理雷达数据
#
# 使用方法：
#   训练：python tools/train.py configs/racformer_with_rhgm_radarbevnet.py
#   测试：python tools/test.py configs/racformer_with_rhgm_radarbevnet.py checkpoints/xxx.pth
# ============================================================

import torch
pi = torch.pi

# ============== 第1部分：数据集基础设置 ==============
# 就像告诉程序"去哪里找数据"

dataset_type = 'CustomNuScenesDataset_radar'
dataset_root = 'data/nuscenes/'  # ⚠️ 如果你的数据在别的地方，这里要改！

input_modality = dict(
    use_lidar=False,    # 不用激光雷达
    use_camera=True,    # 用摄像头 ✅
    use_radar=True,     # 用毫米波雷达 ✅
    use_map=False,      # 不用地图
    use_external=True   # 用额外的数据
)

# 要检测的10个类别（车、卡车、行人等）
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

# ============== 第2部分：空间范围设置 ==============
# 定义"雷达能看多远"（单位：米）

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]  
# 意思是：前后左右各51.2米，高度从-5米到3米

voxel_size = [0.2, 0.2, 8]  
# 把空间切成小格子，每个格子的大小

# ============== 第3部分：模型架构参数 ==============
# 就像搭积木，这些是每个零件的规格

embed_dims = 256        # 特征维度（神经网络的"通道数"）
num_layers = 6          # Transformer的层数
num_frames = 8          # 用几帧历史数据（看8帧以前的画面）
num_levels = 4          # 特征金字塔的层数
num_points = 4          # 采样点数量
num_points_bev = 4      # BEV特征采样点数
img_depth_num = 3       # 图像深度预测层数
bev_depth_num = 5       # BEV深度层数

# 距离区域划分（由近到远）
d_region_list = [0.08, 0.07, 0.06, 0.05, 0.04, 0.03]

# 查询(query)设置：用于检测物体
num_clusters = 6        # 聚类数量
num_ray = 150           # 射线数量
num_query = num_ray * num_clusters  # 总查询数 = 150 * 6 = 900

# ============== 第4部分：数据增强设置 ==============
# 训练时对图片做的"变换"（让模型更鲁棒）

ida_aug_conf = {
    'resize_lim': (0.38, 0.55),      # 图片缩放范围
    'final_dim': (256, 704),         # 最终图片大小
    'bot_pct_lim': (0.0, 0.0),       # 底部裁剪
    'rot_lim': (0.0, 0.0),           # 旋转范围
    'H': 900, 'W': 1600,             # 原始图片大小
    'rand_flip': True,               # 随机翻转
}

# BEV网格配置（鸟瞰图的"地图"格子）
grid_config = {
    'x': [-51.2, 51.2, 0.8],    # X方向：从-51.2到51.2，步长0.8
    'y': [-51.2, 51.2, 0.8],    # Y方向
    'z': [-5, 3, 8],            # Z方向（高度）
    'depth': [1.0, 65.0, 96.0], # 深度范围
    'rcs': [-64, 64, 64]        # RCS（雷达反射强度）范围
}

numC_Trans = 256
file_client_args = dict(backend='disk')

# ============== 第5部分：图像分支模块 ==============
# 处理摄像头图片的"眼睛"

img_backbone = dict(
    type='ResNet',
    depth=50,  # ResNet-50骨干网络
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN2d', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    with_cp=True  # 使用checkpoint节省显存
)

img_neck = dict(
    type='FPN',  # Feature Pyramid Network（特征金字塔）
    in_channels=[256, 512, 1024, 2048],
    out_channels=embed_dims,
    num_outs=num_levels
)

img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],  # ImageNet标准化参数
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

# ============== 🌟 第6部分：新增RHGM模块配置 🌟 ==============
# 这是新加的！用来增强雷达点云

# ⚠️ 重要：变量名改为 rhgm_cfg，与模型代码参数名对应
rhgm_cfg = dict(
    # 核心参数（影响虚拟点生成）
    num_virtual_points=100,      # 每个物体生成100个虚拟点
    dist_thresh=3000,            # 虚拟点距离阈值（单位：mm）
    gauss_sigma=7,               # 高斯分布的"胖瘦"（越大越分散）
    gauss_kernel_size=51,        # 高斯核大小
    gauss_uniform_ratio=[1, 4],  # 高斯采样:均匀采样 = 1:4
    
    # 输入输出设置
    input_channels=7,            # 输入通道数 (x,y,z,rcs,vr,vr_comp,time)
    output_channels=7,           # 输出通道数（保持一致）
    
    # 开关（如果想暂时关闭RHGM，改成False）
    enabled=True,                # ✅ 启用RHGM
)

# 💡 参数调整建议：
# - 如果显存不够：num_virtual_points改成50（减少虚拟点）
# - 如果想要更多细节：num_virtual_points改成200（增加虚拟点）
# - 如果虚拟点太分散：gauss_sigma改小（比如5）
# - 如果虚拟点太集中：gauss_sigma改大（比如10）

# ============== 🌟 第7部分：新增RadarBEVNet模块配置 🌟 ==============
# 这是新加的！用更强大的方法编码雷达特征

# ⚠️ 重要：变量名改为 radar_bev_net_cfg，与模型代码参数名对应
# ⚠️ 注意：feat_channels的最后一个值必须与radar_middle_encoder.in_channels一致（默认64）
radar_bev_net_cfg = dict(
    # 输入参数
    in_channels=7,               # 输入通道数（和RHGM的输出要对应）
    feat_channels=[64],          # ⚠️ 必须是64，与radar_middle_encoder.in_channels匹配！
    
    # 空间参数（要和前面的point_cloud_range对应）
    voxel_size=[0.8, 0.8, 8],
    point_cloud_range=point_cloud_range,
    
    # 高级选项
    with_distance=False,         # 是否用距离特征
    with_pos_embed=True,         # ✅ 使用位置编码（推荐）
    return_rcs=True,             # ✅ 返回RCS特征
    drop=0.0,                    # Dropout概率（防止过拟合）
)

# 💡 参数调整建议：
# - 如果显存不够：feat_channels改成[64]（只用一层）
# - 如果想要更强特征：feat_channels改成[64, 128, 256]（加深网络）
# - 如果训练不稳定：drop改成0.1（增加正则化）

# ============== 第8部分：主模型配置 ==============
# 把所有模块"组装"起来

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
    
    # 🌟 雷达分支模块（使用新模块）🌟
    # 注意：保留原有的voxel_layer用于体素化，但编码器已被替换
    radar_voxel_layer=dict(
        max_num_points=10,              # 每个体素最多10个点
        voxel_size=[0.8, 0.8, 8],       # 体素大小（要和RadarBEVNet对应）
        max_voxels=(30000, 40000),      # 最大体素数（训练/测试）
        point_cloud_range=point_cloud_range,
        deterministic=False,
    ), 
    
    # 🌟 接入RHGM和RadarBEVNet 🌟
    # ⚠️ 重要：参数名必须与模型代码中的 __init__ 方法参数名一致
    use_rhgm=True,                      # ✅ 启用RHGM模块
    rhgm_cfg=rhgm_cfg,                  # RHGM配置（雷达点云增强模块）
    use_radar_bev_net=True,             # ✅ 启用RadarBEVNet模块
    radar_bev_net_cfg=radar_bev_net_cfg,  # RadarBEVNet配置（雷达BEV特征编码模块）
    
    # ⚠️ 注意：即使使用RadarBEVNet，仍需保留原有编码器配置（模型代码需要）
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
        in_channels=64, 
        output_shape=(128, 128)
    ),
    
    # 检测头（保持不变）
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
        # ⚠️ 关键修复：显式传递pc_range参数（问题1修复）
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

# ============== 第9部分：数据处理流程 ==============
# 训练时怎么读取和处理数据

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

# 测试时的数据处理流程（不做增强）
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1, test_mode=True),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='Loadnuradarpoints', coord_type='RADAR', num_sweeps=5, file_client_args=file_client_args),
    dict(type='LoadradarpointsFromMultiSweeps', sweeps_num=num_frames-1, num_aggr_sweeps=5, test_mode=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='RadarPointToMultiViewDepth', downsample=1, grid_config=grid_config, test_mode=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RaCFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['img', 'gt_depth', 'radar_points', 'radar_depth', 'radar_rcs'], meta_keys=(
                'filename', 'box_type_3d', 'ori_shape', 'img_shape', 'pad_shape',
                'lidar2img', 'img_timestamp', 'intrinsics'))
        ])
]

# ============== 第10部分：数据集配置 ==============
# 告诉程序去哪里找训练/验证/测试数据

data = dict(
    workers_per_gpu=4,  # 每个GPU用4个线程读数据
    
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_train_sweep.pkl',  # 训练集标注文件
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'
    ),
    
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_val_sweep.pkl',  # 验证集标注文件
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'
    ),
    
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_test_sweep.pkl',  # 测试集标注文件
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'
    )
)

# ============== 第11部分：优化器配置 ==============
# 控制模型怎么"学习"

optimizer = dict(
    type='AdamW',       # 优化器类型（AdamW比较稳定）
    lr=4e-4,            # 学习率（控制学习速度）
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),      # 图像骨干网络学得慢一点
        'sampling_offset': dict(lr_mult=0.1),   # 采样偏移学得慢一点
    }),
    weight_decay=0.01   # 权重衰减（防止过拟合）
)

# 💡 调整建议：
# - 如果训练不收敛：lr改小（比如2e-4）
# - 如果收敛太慢：lr改大（比如8e-4）

optimizer_config = dict(
    type='Fp16OptimizerHook',         # 使用FP16混合精度（省显存，加速）
    loss_scale=512.0,                 # 损失缩放（防止数值溢出）
    grad_clip=dict(max_norm=35, norm_type=2)  # 梯度裁剪（防止梯度爆炸）
)

# ============== 第12部分：学习率策略 ==============
# 控制学习率怎么变化

lr_config = dict(
    policy='CosineAnnealing',  # 余弦退火策略（先快后慢）
    warmup='linear',           # 前期线性预热
    warmup_iters=500,          # 预热500次迭代
    warmup_ratio=1.0 / 3,      # 预热阶段学习率是最大学习率的1/3
    min_lr_ratio=1e-3          # 最小学习率是最大学习率的1/1000
)

# ============== 第13部分：训练参数 ==============

total_epochs = 20   # 总共训练36轮（1轮=看完整个数据集1遍）
batch_size = 4      # 每次喂给模型2个样本

# 💡 显存不够？batch_size改成1
# 💡 想快点训练？如果有多张卡，可以改大batch_size

# ============== 第14部分：预训练权重 ==============
# 从哪里加载预训练模型（让训练"站在巨人肩膀上"）

load_from = 'pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
revise_keys = [('backbone', 'img_backbone')]

# ⚠️ 如果没有这个文件，需要下载或注释掉这行

# 恢复训练（如果中断了）
resume_from = None  # 如果要继续上次的训练，填checkpoint路径

# ============== 第15部分：检查点保存 ==============
# 每隔多久保存一次模型

default_hooks = dict(
    checkpoint = None
)

checkpoint_config = dict(
    interval=1,          # 每1个epoch保存一次
    max_keep_ckpts=4     # 最多保留4个checkpoint（省硬盘空间）
)

# ============== 第16部分：日志记录 ==============
# 训练过程中打印什么信息

log_config = dict(
    interval=1,
    hooks=[
        dict(type='MyTextLoggerHook', interval=50, reset_flag=True),         # 文本日志
        dict(type='MyTensorboardLoggerHook', interval=500, reset_flag=True)  # TensorBoard可视化
    ]
)

# ============== 第17部分：评估配置 ==============

eval_config = dict(interval=2)  # 每2个epoch评估一次

# ============== 第18部分：其他设置 ==============

debug = False  # 如果要调试，改成True

custom_hooks = [
    dict(
        type='SequentialControlHook',
        start_epoch=18,  # 从第18轮开始执行某些特殊操作
    ),
]

# ============================================================
# 🎉 配置文件结束！
# ============================================================
# 
# 快速参考：
# 
# 【重要参数】
# - dataset_root: 数据集路径（第1部分）
# - num_virtual_points: RHGM虚拟点数量（第6部分）
# - feat_channels: RadarBEVNet特征通道（第7部分）
# - batch_size: 批量大小（第13部分）
# - total_epochs: 训练轮数（第13部分）
# - lr: 学习率（第11部分）
# 
# 【显存不够怎么办】
# 1. batch_size改成1
# 2. num_virtual_points改成50
# 3. feat_channels改成[64]
# 4. num_frames改成4（减少历史帧）
# 
# 【想要更好效果】
# 1. total_epochs改成48或60
# 2. num_virtual_points改成200
# 3. feat_channels改成[64, 128, 256]
# 
# 有问题？看下面的"运行指南.md"！
# ============================================================

