_base_ = [
    '../_base_/datasets/STAR_sub8k_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['UHR_DETR'], allow_failed_imports=False)

# pretrained = 'https://github.com/flytocc/mmdetection/releases/download/model_zoo/resnet50vd_ssld_v2_pretrained_edfe4074.pth'  # noqa
# pretrained = '/path/to/your/pretrained/checkpoint'
pretrained = '/data/ljf/checkpoints/ResNet/resnet50vd_ssld_v2_pretrained_edfe4074.pth'


globalNet_cfg = dict(
    backbone=dict(
        type='ResNet',
        with_cp=True,
        depth=18,
        out_indices=(1, 2, 3,),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[128, 256, 512],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        num_outs=3
    ),
    map_head_loss=dict(type='DistributionFocalLoss', loss_weight=1.0),
)

model = dict(
    type='UHR_DETR',
    globalNet_cfg=globalNet_cfg,
    num_queries=1000,
    min_num_query=300,  # min
    max_num_query=3000, # max
    num_query_expansion_ratio=1.5,
    top_k_patch=40,
    top_k_patch_infer=40,
    patch_size=(512, 512),
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNetV1d',  # ResNet for DINO
        with_cp=True,
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=0,  # -1 for DINO
        norm_cfg=dict(type='BN', requires_grad=False),  # BN for DINO
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),  # GN for DINO
        num_outs=3),  # 4 for DINO
    encoder=dict(
        use_encoder_idx=[-1],
        num_encoder_layers=1,
        in_channels=[256, 256, 256],
        fpn_cfg=dict(
            type='RTDETRFPN',
            in_channels=[256, 256, 256],
            out_channels=256,
            expansion=1.0,
            norm_cfg=dict(type='BN', requires_grad=True)),
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,  # 2048 for DINO
                ffn_drop=0.0,
                act_cfg=dict(type='GELU')))),  # ReLU for DINO
    decoder=dict(
        with_checkpoint=True,
        num_layers=6,
        num_heads=8,
        num_points=4,
        ffn_channels=1024,
        dropout=0.0),
    bbox_head=dict(
        type='RTDETRHead',
        num_classes=8,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='RTDETRVarifocalLoss',  # FocalLoss in DINO
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=3000),
)

train_dataloader = dict(
    batch_size=1,
    num_workers=2,)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1)},
        norm_decay_mult=0,
        bypass_duplicate=True))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=20),
    visualization=dict(type='DetVisualizationHook', score_thr=0.10))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)

custom_hooks = [
    dict(type='EmptyCacheHook', after_epoch=True, after_iter=False)
]
