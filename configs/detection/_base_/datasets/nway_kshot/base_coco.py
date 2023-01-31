# dataset settings
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_multi_pipelines = dict(
    query=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ],
    support=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='GenerateMask', target_size=(224, 224)),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# classes splits are predefined in FewShotCocoDataset
data_root = 'data/coco/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='NWayKShotDataset',
        num_support_ways=60,
        num_support_shots=1,
        one_support_shot_per_image=True,
        num_used_support_shots=200,
        save_dataset=False,
        dataset=dict(
            type='FewShotCocoDataset',
            ann_cfg=[
                dict(
                    type='ann_file',
                    ann_file='data/few_shot_ann/coco/annotations/train.json')
            ],
            img_prefix=data_root,
            multi_pipelines=train_multi_pipelines,
            classes='BASE_CLASSES',
            instance_wise=False,
            dataset_name='query_support_dataset')),
    val=dict(
        type='FewShotCocoDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/few_shot_ann/coco/annotations/val.json')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes='BASE_CLASSES'),
    test=dict(
        type='FewShotCocoDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/few_shot_ann/coco/annotations/val.json')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        classes='BASE_CLASSES'),
    model_init=dict(
        copy_from_train_dataset=True,
        samples_per_gpu=16,
        workers_per_gpu=1,
        type='FewShotCocoDataset',
        ann_cfg=None,
        img_prefix=data_root,
        pipeline=train_multi_pipelines['support'],
        instance_wise=True,
        classes='BASE_CLASSES',
        dataset_name='model_init_dataset'))
evaluation = dict(interval=20000, metric='bbox', classwise=True)
