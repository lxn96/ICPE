# Copyright (c) OpenMMLab. All rights reserved.

import copy
from os.path import dirname, exists, join

import numpy as np
import pytest
import torch
from mmcv.runner import wrap_fp16_model

from mmfewshot.detection.models import build_detector


def _get_config_directory():
    """Find the predefined detector config directory."""
    try:
        # Assume we are running in the source mmfewshot repo
        repo_dpath = dirname(dirname(dirname(dirname(__file__))))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmfewshot
        repo_dpath = dirname(dirname(mmfewshot.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def _get_detector_cfg(fname):
    """Grab configs necessary to create a detector.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    return model


def _demo_mm_inputs(input_shape=(1, 3, 300, 300),
                    num_items=None, num_classes=10):  # yapf: disable
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_items (None | List[int]):
            specifies the number of boxes in each batch item
        num_classes (int):
            number of different labels a box might have
    """
    from mmdet.core import BitmapMasks

    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)

    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': np.array([1.1, 1.2, 1.1, 1.2]),
        'flip': False,
        'flip_direction': None,
    } for _ in range(N)]

    gt_bboxes = []
    gt_labels = []
    gt_masks = []

    for batch_idx in range(N):
        if num_items is None:
            num_boxes = rng.randint(1, 10)
        else:
            num_boxes = num_items[batch_idx]

        cx, cy, bw, bh = rng.rand(num_boxes, 4).T

        tl_x = ((cx * W) - (W * bw / 2)).clip(0, W)
        tl_y = ((cy * H) - (H * bh / 2)).clip(0, H)
        br_x = ((cx * W) + (W * bw / 2)).clip(0, W)
        br_y = ((cy * H) + (H * bh / 2)).clip(0, H)

        boxes = np.vstack([tl_x, tl_y, br_x, br_y]).T
        class_idxs = rng.randint(1, num_classes, size=num_boxes)

        gt_bboxes.append(torch.FloatTensor(boxes))
        gt_labels.append(torch.LongTensor(class_idxs))

    mask = np.random.randint(0, 2, (len(boxes), H, W), dtype=np.uint8)
    gt_masks.append(BitmapMasks(mask, H, W))

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'img_metas': img_metas,
        'gt_bboxes': gt_bboxes,
        'gt_labels': gt_labels,
        'gt_bboxes_ignore': None,
        'gt_masks': gt_masks,
    }

    return mm_inputs


@pytest.mark.parametrize(
    'cfg_file',
    [
        'detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_base-training.py',  # noqa
        'detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_1shot-fine-tuning.py'  # noqa
    ])
def test_mpsr_detector_forward(cfg_file):
    if not torch.cuda.is_available():
        import pytest
        pytest.skip('test requires GPU and torch+cuda')
    model = _get_detector_cfg(cfg_file)
    model.backbone.init_cfg = None
    model = build_detector(model)
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    # test train forward
    main_data = dict(
        img_metas=[{
            'img_shape': (256, 256, 3),
            'scale_factor': 1,
            'pad_shape': (256, 256, 3)
        }],
        img=torch.randn(1, 3, 256, 256),
        gt_bboxes=copy.deepcopy(gt_bboxes),
        gt_labels=[torch.LongTensor([0])],
        query_class=torch.LongTensor([0]))
    auxiliary_data = dict()
    scales = [32, 64, 128, 256, 512, 800]
    for i in range(6):
        auxiliary_data.update({
            f'img_metas_scale_{i}': [{
                'img_shape': (scales[i], scales[i], 3),
                'scale_factor': 1,
                'pad_shape': (scales[i], scales[i], 3)
            }],
            f'img_scale_{i}':
            torch.randn(1, 3, scales[i], scales[i]),
            f'gt_bboxes_scale_{i}':
            copy.deepcopy(gt_bboxes),
            f'gt_labels_scale_{i}': [torch.LongTensor([0])]
        })
    model.train()
    losses = model(
        main_data=main_data, auxiliary_data=auxiliary_data, return_loss=True)
    assert 'loss_rpn_cls' in losses
    assert 'loss_rpn_bbox' in losses
    assert 'loss_cls' in losses
    assert 'acc' in losses
    assert 'loss_bbox' in losses
    assert 'loss_rpn_cls_auxiliary' in losses
    assert 'loss_cls_auxiliary' in losses
    assert 'acc_auxiliary' in losses


@pytest.mark.parametrize(
    'cfg_file',
    [
        'detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_base-training.py',  # noqa
        'detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_1shot-fine-tuning.py'  # noqa
    ])
def test_mpsr_detector_fp16_forward(cfg_file):
    if not torch.cuda.is_available():
        import pytest
        pytest.skip('test requires GPU and torch+cuda')
    model = _get_detector_cfg(cfg_file)
    model.backbone.init_cfg = None
    model = build_detector(model)
    wrap_fp16_model(model)
    model = model.cuda()
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]).cuda(),
    ]
    # test train forward
    main_data = dict(
        img_metas=[{
            'img_shape': (256, 256, 3),
            'scale_factor': 1,
            'pad_shape': (256, 256, 3)
        }],
        img=torch.randn(1, 3, 256, 256).cuda(),
        gt_bboxes=copy.deepcopy(gt_bboxes),
        gt_labels=[torch.LongTensor([0]).cuda()],
        query_class=torch.LongTensor([0]).cuda())
    auxiliary_data = dict()
    scales = [32, 64, 128, 256, 512, 800]
    for i in range(6):
        auxiliary_data.update({
            f'img_metas_scale_{i}': [{
                'img_shape': (scales[i], scales[i], 3),
                'scale_factor': 1,
                'pad_shape': (scales[i], scales[i], 3)
            }],
            f'img_scale_{i}':
            torch.randn(1, 3, scales[i], scales[i]).cuda(),
            f'gt_bboxes_scale_{i}':
            copy.deepcopy(gt_bboxes),
            f'gt_labels_scale_{i}': [torch.LongTensor([0]).cuda()]
        })
    model.train()
    losses = model(
        main_data=main_data, auxiliary_data=auxiliary_data, return_loss=True)
    assert 'loss_rpn_cls' in losses
    assert 'loss_rpn_bbox' in losses
    assert 'loss_cls' in losses
    assert 'acc' in losses
    assert 'loss_bbox' in losses
    assert 'loss_rpn_cls_auxiliary' in losses
    assert 'loss_cls_auxiliary' in losses
    assert 'acc_auxiliary' in losses
