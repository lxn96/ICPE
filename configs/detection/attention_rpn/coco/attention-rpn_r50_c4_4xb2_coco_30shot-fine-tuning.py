_base_ = [
    '../../_base_/datasets/query_aware/few_shot_coco.py',
    '../../_base_/schedules/schedule.py', '../attention-rpn_r50_c4.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
num_support_ways = 2
num_support_shots = 10
data = dict(
    train=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        repeat_times=50,
        dataset=dict(
            type='FewShotCocoDefaultDataset',
            ann_cfg=[dict(method='Attention_RPN', setting='30SHOT')],
            num_novel_shots=30,
            classes='NOVEL_CLASSES',
            min_bbox_area=0,
            instance_wise=False)),
    val=dict(classes='NOVEL_CLASSES'),
    test=dict(classes='NOVEL_CLASSES'),
    model_init=dict(classes='NOVEL_CLASSES'))
evaluation = dict(interval=3000)
checkpoint_config = dict(interval=3000)
optimizer = dict(
    lr=0.001,
    momentum=0.9,
    paramwise_cfg=dict(custom_keys={'roi_head.bbox_head': dict(lr_mult=2.0)}))
lr_config = dict(
    warmup_iters=400, warmup_ratio=0.1, step=[
        3000,
        5000,
    ])
log_config = dict(interval=10)
runner = dict(max_iters=6000)
# load_from = 'path of base training model'
load_from = 'work_dirs/attention-rpn_r50_c4_4xb2_coco_base-training/latest.pth'
model = dict(
    frozen_parameters=['backbone'],
    rpn_head=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
    ),
    roi_head=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
    ),
)
