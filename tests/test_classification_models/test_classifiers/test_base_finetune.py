# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest
import torch

from mmfewshot.classification.models import CLASSIFIERS


@pytest.mark.parametrize('classifier',
                         ['Baseline', 'BaselinePlus', 'NegMargin'])
def test_image_classifier(classifier):
    model_cfg = dict(type=classifier, backbone=dict(type='Conv4'))

    imgs = torch.randn(4, 3, 84, 84)
    feats = torch.randn(4, 1600)
    label = torch.LongTensor([0, 1, 2, 3])

    model_cfg_ = copy.deepcopy(model_cfg)
    model = CLASSIFIERS.build(model_cfg_)

    # test property
    assert not model.with_neck
    assert model.with_head

    assert model.device
    assert model.get_device()

    # test train_step
    outputs = model.train_step(
        {
            'img': imgs,
            'gt_label': label,
            'mode': 'train',
            'img_metas': [_ for _ in range(4)]
        }, None)
    assert outputs['loss'].item() > 0
    assert outputs['num_samples'] == 4

    # test train_step
    outputs = model.train_step(
        {
            'feats': feats,
            'gt_label': label,
            'mode': 'train',
            'img_metas': [_ for _ in range(4)]
        }, None)
    assert outputs['loss'].item() > 0
    assert outputs['num_samples'] == 4

    # test extract features
    outputs = model(**{'img': imgs, 'gt_label': label, 'mode': 'extract_feat'})
    assert outputs.size(0) == 4

    model.before_meta_test(dict())
    model.before_forward_support()

    # test support step
    outputs = model(**{'img': imgs, 'gt_label': label, 'mode': 'support'})
    assert outputs['loss'].item() > 0

    # test support step
    outputs = model(**{'feats': feats, 'gt_label': label, 'mode': 'support'})
    assert outputs['loss'].item() > 0

    model.before_forward_query()
    # test query step
    outputs = model(**{'img': imgs, 'mode': 'query'})
    assert isinstance(outputs, list)
    assert len(outputs) == 4
    assert outputs[0].shape[0] == 5

    # test query step
    outputs = model(**{'feats': feats, 'mode': 'query'})
    assert isinstance(outputs, list)
    assert len(outputs) == 4
    assert outputs[0].shape[0] == 5

    with pytest.raises(ValueError):
        # test extract features
        model(**{'img': imgs, 'gt_label': label, 'mode': 'test'})
