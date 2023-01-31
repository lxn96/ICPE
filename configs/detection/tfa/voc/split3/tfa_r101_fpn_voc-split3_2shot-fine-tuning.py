_base_ = [
    '../../../_base_/datasets/fine_tune_based/few_shot_voc.py',
    '../../../_base_/schedules/schedule.py', '../../tfa_r101_fpn.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        type='FewShotVOCDefaultDataset',
        ann_cfg=[dict(method='TFA', setting='SPLIT3_2SHOT')],
        num_novel_shots=2,
        num_base_shots=2,
        classes='ALL_CLASSES_SPLIT3'),
    val=dict(classes='ALL_CLASSES_SPLIT3'),
    test=dict(classes='ALL_CLASSES_SPLIT3'))
evaluation = dict(
    interval=8000,
    class_splits=['BASE_CLASSES_SPLIT3', 'NOVEL_CLASSES_SPLIT3'])
checkpoint_config = dict(interval=8000)
optimizer = dict(lr=0.001)
lr_config = dict(
    warmup=None, step=[
        7000,
    ])
runner = dict(max_iters=8000)
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/tfa/README.md for more details.
load_from = ('work_dirs/tfa_r101_fpn_voc-split3_base-training/'
             'base_model_random_init_bbox_head.pth')
