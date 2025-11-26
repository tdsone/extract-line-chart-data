model = dict(
    type="CascadeRCNN",
    backbone=dict(
        type="SwinTransformer",
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
        ),
    ),
    neck=dict(
        type="FPN", in_channels=[96, 192, 384, 768], out_channels=256, num_outs=5
    ),
    rpn_head=dict(
        type="RPNHead",
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="SmoothL1Loss", beta=0.1111111111111111, loss_weight=1.0),
    ),
    roi_head=dict(
        type="CascadeRoIHead_LGF",
        num_stages=3,
        stage_loss_weights=[1, 1, 0.5],
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=[
            dict(
                type="Shared3FCBBoxHead_with_BboxEncoding",
                in_channels=256,
                fc_out_channels=1024,
                bbox_encoding_dim=512,
                roi_feat_size=7,
                num_classes=18,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type="FocalLoss"),
                loss_bbox=dict(type="BalancedL1Loss", beta=1.0, loss_weight=1.0),
            ),
            dict(
                type="Shared3FCBBoxHead_with_BboxEncoding",
                in_channels=256,
                fc_out_channels=1024,
                bbox_encoding_dim=512,
                roi_feat_size=7,
                num_classes=18,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type="FocalLoss"),
                loss_bbox=dict(type="BalancedL1Loss", beta=1.0, loss_weight=1.0),
            ),
            dict(
                type="Shared3FCBBoxHead_with_BboxEncoding",
                in_channels=256,
                fc_out_channels=1024,
                bbox_encoding_dim=512,
                roi_feat_size=7,
                num_classes=18,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type="FocalLoss"),
                loss_bbox=dict(type="BalancedL1Loss", beta=1.0, loss_weight=1.0),
            ),
        ],
        localglobal_fuser=dict(
            type="LocalGlobal_Context_Fuser",
            channels=256,
            roi_size=7,
            reduced_channels=256,
            lg_merge_layer=dict(type="SELayer", channels=256),
        ),
        lgf_shared=False,
        bbox_encoder=dict(
            type="BboxEncoder",
            n_layer=4,
            n_head=4,
            n_embd=512,
            bbox_cord_dim=4,
            bbox_max_num=1024,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        ),
        bbox_encoder_shared=False,
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=0,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=[
            dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True,
                ),
                pos_weight=-1,
                debug=False,
            ),
            dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True,
                ),
                pos_weight=-1,
                debug=False,
            ),
            dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True,
                ),
                pos_weight=-1,
                debug=False,
            ),
        ],
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.0, nms=dict(type="nms", iou_threshold=0.7), max_per_img=200
        ),
    ),
)
dataset_type = "CocoDataset"
data_root = "data/coco/"
img_norm_cfg = dict(
    mean=[216.45, 212.36, 206.76], std=[55.82, 56.04, 55.56], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="AutoAugment",
        policies=[
            [
                {
                    "type": "Resize",
                    "img_scale": [
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    "multiscale_mode": "value",
                    "keep_ratio": True,
                }
            ],
            [
                {
                    "type": "Resize",
                    "img_scale": [(400, 1333), (500, 1333), (600, 1333)],
                    "multiscale_mode": "value",
                    "keep_ratio": True,
                },
                {
                    "type": "RandomCrop",
                    "crop_type": "absolute_range",
                    "crop_size": (384, 600),
                    "allow_negative_crop": True,
                },
                {
                    "type": "Resize",
                    "img_scale": [
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    "multiscale_mode": "value",
                    "override": True,
                    "keep_ratio": True,
                },
                {
                    "type": "PhotoMetricDistortion",
                    "brightness_delta": 32,
                    "contrast_range": (0.5, 1.5),
                    "saturation_range": (0.5, 1.5),
                    "hue_delta": 18,
                },
                {
                    "type": "MinIoURandomCrop",
                    "min_ious": (0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                    "min_crop_size": 0.3,
                },
                {
                    "type": "CutOut",
                    "n_holes": (5, 10),
                    "cutout_shape": [
                        (4, 4),
                        (4, 8),
                        (8, 4),
                        (8, 8),
                        (16, 32),
                        (32, 16),
                        (32, 32),
                        (32, 48),
                        (48, 32),
                        (48, 48),
                    ],
                },
            ],
        ],
    ),
    dict(type="RandomFlip", flip_ratio=0.1),
    dict(
        type="Normalize",
        mean=[216.45, 212.36, 206.76],
        std=[55.82, 56.04, 55.56],
        to_rgb=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip", flip_ratio=0.0),
            dict(
                type="Normalize",
                mean=[216.45, 212.36, 206.76],
                std=[55.82, 56.04, 55.56],
                to_rgb=True,
            ),
            dict(type="Pad", size_divisor=32),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=4,
    train=dict(
        type="CocoDataset",
        ann_file="./data/pmc_2022/pmc_coco/element_detection/train.json",
        img_prefix="./data/pmc_2022/pmc_coco/element_detection/train/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                type="AutoAugment",
                policies=[
                    [
                        {
                            "type": "Resize",
                            "img_scale": [
                                (480, 1333),
                                (512, 1333),
                                (544, 1333),
                                (576, 1333),
                                (608, 1333),
                                (640, 1333),
                                (672, 1333),
                                (704, 1333),
                                (736, 1333),
                                (768, 1333),
                                (800, 1333),
                            ],
                            "multiscale_mode": "value",
                            "keep_ratio": True,
                        }
                    ],
                    [
                        {
                            "type": "Resize",
                            "img_scale": [(400, 1333), (500, 1333), (600, 1333)],
                            "multiscale_mode": "value",
                            "keep_ratio": True,
                        },
                        {
                            "type": "RandomCrop",
                            "crop_type": "absolute_range",
                            "crop_size": (384, 600),
                            "allow_negative_crop": True,
                        },
                        {
                            "type": "Resize",
                            "img_scale": [
                                (480, 1333),
                                (512, 1333),
                                (544, 1333),
                                (576, 1333),
                                (608, 1333),
                                (640, 1333),
                                (672, 1333),
                                (704, 1333),
                                (736, 1333),
                                (768, 1333),
                                (800, 1333),
                            ],
                            "multiscale_mode": "value",
                            "override": True,
                            "keep_ratio": True,
                        },
                        {
                            "type": "PhotoMetricDistortion",
                            "brightness_delta": 32,
                            "contrast_range": (0.5, 1.5),
                            "saturation_range": (0.5, 1.5),
                            "hue_delta": 18,
                        },
                        {
                            "type": "MinIoURandomCrop",
                            "min_ious": (0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                            "min_crop_size": 0.3,
                        },
                        {
                            "type": "CutOut",
                            "n_holes": (5, 10),
                            "cutout_shape": [
                                (4, 4),
                                (4, 8),
                                (8, 4),
                                (8, 8),
                                (16, 32),
                                (32, 16),
                                (32, 32),
                                (32, 48),
                                (48, 32),
                                (48, 48),
                            ],
                        },
                    ],
                ],
            ),
            dict(type="RandomFlip", flip_ratio=0.1),
            dict(
                type="Normalize",
                mean=[216.45, 212.36, 206.76],
                std=[55.82, 56.04, 55.56],
                to_rgb=True,
            ),
            dict(type="Pad", size_divisor=32),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
        ],
        classes=[
            "x_title",
            "y_title",
            "plot_area",
            "other",
            "xlabel",
            "ylabel",
            "chart_title",
            "x_tick",
            "y_tick",
            "legend_patch",
            "legend_label",
            "legend_title",
            "legend_area",
            "mark_label",
            "value_label",
            "y_axis_area",
            "x_axis_area",
            "tick_grouping",
        ],
    ),
    val=dict(
        type="CocoDataset",
        ann_file="./data/pmc_2022/pmc_coco/element_detection/val.json",
        img_prefix="./data/pmc_2022/pmc_coco/element_detection/val/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(
                        type="Normalize",
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True,
                    ),
                    dict(type="Pad", size_divisor=32),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
        classes=[
            "x_title",
            "y_title",
            "plot_area",
            "other",
            "xlabel",
            "ylabel",
            "chart_title",
            "x_tick",
            "y_tick",
            "legend_patch",
            "legend_label",
            "legend_title",
            "legend_area",
            "mark_label",
            "value_label",
            "y_axis_area",
            "x_axis_area",
            "tick_grouping",
        ],
    ),
    test=dict(
        type="CocoDataset",
        ann_file="./data/pmc_2022/pmc_coco/element_detection/split3_test.json",
        img_prefix="./data/pmc_2022/pmc_coco/element_detection/split3_test/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(
                        type="Normalize",
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True,
                    ),
                    dict(type="Pad", size_divisor=32),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
        classes=[
            "x_title",
            "y_title",
            "plot_area",
            "other",
            "xlabel",
            "ylabel",
            "chart_title",
            "x_tick",
            "y_tick",
            "legend_patch",
            "legend_label",
            "legend_title",
            "legend_area",
            "mark_label",
            "value_label",
            "y_axis_area",
            "x_axis_area",
            "tick_grouping",
        ],
    ),
)
evaluation = dict(interval=1, metric=["bbox"])
optimizer = dict(
    type="AdamW",
    lr=0.0002,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
        )
    ),
)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11]
)
runner = dict(type="EpochBasedRunner", max_epochs=150)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])
custom_hooks = [dict(type="NumClassCheckHook")]
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
opencv_num_threads = 0
mp_start_method = "fork"
auto_scale_lr = dict(enable=False, base_batch_size=16)
pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"
classes = [
    "x_title",
    "y_title",
    "plot_area",
    "other",
    "xlabel",
    "ylabel",
    "chart_title",
    "x_tick",
    "y_tick",
    "legend_patch",
    "legend_label",
    "legend_title",
    "legend_area",
    "mark_label",
    "value_label",
    "y_axis_area",
    "x_axis_area",
    "tick_grouping",
]
auto_resume = False
gpu_ids = range(0, 4)
