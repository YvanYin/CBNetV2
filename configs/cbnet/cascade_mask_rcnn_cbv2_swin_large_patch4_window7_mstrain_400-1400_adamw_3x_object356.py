_base_ = [
<<<<<<< HEAD
    '../swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_object365.py'
=======
    '../swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'
>>>>>>> f7c8f2ed6ca7ba9379074410abca4886c6e69cdc
]

model = dict(
    backbone=dict(
        type='CBSwinTransformer',
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False
    ),
    neck=dict(
        type='CBFPN',
        in_channels=[192, 384, 768, 1536]

    ),
    test_cfg = dict(
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='soft_nms'),
        )
    )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from HTC
data_root="datasets/objects365/"
data = dict(
    train=dict(
<<<<<<< HEAD
        ann_file=data_root + 'annotations/instances_train.json',
=======
        ann_file=data_root + 'annotations/train_annotations.json',
>>>>>>> f7c8f2ed6ca7ba9379074410abca4886c6e69cdc
        img_prefix=data_root + 'train/'))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Resize',
        # img_scale=[(1600, 400), (1600, 1400)],
        img_scale=[(256,256), (256,256)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 1400),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
samples_per_gpu = 1
data = dict(samples_per_gpu=samples_per_gpu,
            train=dict(pipeline=train_pipeline),
            val=dict(pipeline=test_pipeline),
            test=dict(pipeline=test_pipeline))
optimizer = dict(lr=0.0001*(samples_per_gpu/2))
