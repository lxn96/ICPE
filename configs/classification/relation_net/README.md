# Learning to Compare: Relation Network for Few-Shot Learning <a href="https://arxiv.org/abs/1711.06025"> (CVPR'2018)</a>

## Abstract

<!-- [ABSTRACT] -->

We present a conceptually simple, flexible, and general
framework for few-shot learning, where a classifier must
learn to recognise new classes given only few examples from
each. Our method, called the Relation Network (RN), is
trained end-to-end from scratch. During meta-learning, it
learns to learn a deep distance metric to compare a small number of
images within episodes, each of which is designed to simulate the
few-shot setting. Once trained, a RN is able to classify images of new classes by computing relation scores between query images and the few examples of
each new class without further updating the network.
Besides providing improved performance on few-shot learning, our framework is easily extended to zero-shot learning.
Extensive experiments on five benchmarks demonstrate that
our simple approach provides a unified and effective approach for both of these two tasks.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/15669896/142848503-5a97747b-7462-4d52-b4b0-7697c452a973.png" width="80%"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{sung2018learning,
    title={Learning to compare: Relation network for few-shot learning},
    author={Sung, Flood and Yang, Yongxin and Zhang, Li and Xiang, Tao and Torr, Philip HS and Hospedales, Timothy M},
    booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
    pages={1199--1208},
    year={2018}
}
```

## How to Reproduce RelationNet

It consists of two steps:

- **Step1: Base training**

  - use all the images of base classes to train a base model.
  - conduct meta testing on validation set to select the best model.

- **Step2: Meta Testing**:

  - use best model from step1, the best model are saved into `${WORK_DIR}/${CONFIG}/best_accuracy_mean.pth` in default.

### An example of CUB dataset with Conv4

```bash
# base training
python ./tools/classification/train.py \
  configs/classification/relation_net/cub/relation-net_conv4_1xb105_cub_5way-1shot.py

# meta testing
python ./tools/classification/test.py \
  configs/classification/relation_net/cub/relation-net_conv4_1xb105_cub_5way-1shot.py \
  work_dir/relation-net_conv4_1xb105_cub_5way-1shot/best_accuracy_mean.pth
```

**Note**:

- All the result are trained with single gpu.
- The config of 1 shot and 5 shot use same training setting,
  but different meta test setting on validation set and test set.
- Currently, we use model selected by 1 shot validation (100 episodes) to
  evaluate both 1 shot and 5 shot setting on test set.
- The hyper-parameters in configs are roughly set and probably not the optimal one so
  feel free to tone and try different configurations.
  For example, try different learning rate or validation episodes for each setting.
  Anyway, we will continue to improve it.
- The training batch size is calculated by `num_support_way` * (`num_support_shots` + `num_query_shots`)

## Results on CUB dataset with 2000 episodes

| Arch                                                                                                | Input Size | Batch Size | way | shot | mean Acc | std  |                                                                           ckpt                                                                            |                                                                 log                                                                  |
| :-------------------------------------------------------------------------------------------------- | :--------: | :--------: | :-: | :--: | :------: | :--: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: |
| [conv4](/configs/classification/relation_net/cub/relation-net_conv4_1xb105_cub_5way-1shot.py)       |   84x84    |    105     |  5  |  1   |  65.53   | 0.51 |  [ckpt](https://download.openmmlab.com/mmfewshot/classification/relation_net/cub/relation-net_conv4_1xb105_cub_5way-1shot_20211120_101425-63bfa087.pth)   |  [log](https://download.openmmlab.com/mmfewshot/classification/relation_net/cub/relation-net_conv4_1xb105_cub_5way-1shot.log.json)   |
| [conv4](/configs/classification/relation_net/cub/relation-net_conv4_1xb105_cub_5way-5shot.py)       |   84x84    |    105     |  5  |  5   |  82.05   | 0.34 |                                                                             ⇑                                                                             |                                                                  ⇑                                                                   |
| [resnet12](/configs/classification/relation_net/cub/relation-net_resnet12_1xb105_cub_5way-1shot.py) |   84x84    |    105     |  5  |  1   |  73.22   | 0.48 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/relation_net/cub/relation-net_resnet12_1xb105_cub_5way-1shot_20211120_101425-d67efa62.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/relation_net/cub/relation-net_resnet12_1xb105_cub_5way-1shot.log.json) |
| [resnet12](/configs/classification/relation_net/cub/relation-net_resnet12_1xb105_cub_5way-5shot.py) |   84x84    |    105     |  5  |  5   |  86.94   | 0.28 |                                                                             ⇑                                                                             |                                                                  ⇑                                                                   |

## Results on Mini-ImageNet dataset with 2000 episodes

| Arch                                                                                                                    | Input Size | Batch Size | way | shot | mean Acc | std  |                                                                                     ckpt                                                                                      |                                                                           log                                                                            |
| :---------------------------------------------------------------------------------------------------------------------- | :--------: | :--------: | :-: | :--: | :------: | :--: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [conv4](/configs/classification/relation_net/mini_imagenet/relation-net_conv4_1xb105_mini-imagenet_5way-1shot.py)       |   84x84    |    105     |  5  |  1   |  49.69   | 0.43 |  [ckpt](https://download.openmmlab.com/mmfewshot/classification/relation_net/mini_imagenet/relation-net_conv4_1xb105_mini-imagenet_5way-1shot_20211120_105359-c4256d35.pth)   |  [log](https://download.openmmlab.com/mmfewshot/classification/relation_net/mini_imagenet/relation-net_conv4_1xb105_mini-imagenet_5way-1shot.log.json)   |
| [conv4](/configs/classification/relation_net/mini_imagenet/relation-net_conv4_1xb105_mini-imagenet_5way-5shot.py)       |   84x84    |    105     |  5  |  5   |  68.14   | 0.37 |                                                                                       ⇑                                                                                       |                                                                            ⇑                                                                             |
| [resnet12](/configs/classification/relation_net/mini_imagenet/relation-net_resnet12_1xb105_mini-imagenet_5way-1shot.py) |   84x84    |    105     |  5  |  1   |  54.12   | 0.46 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/relation_net/mini_imagenet/relation-net_resnet12_1xb105_mini-imagenet_5way-1shot_20211120_110716-c73a3f81.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/relation_net/mini_imagenet/relation-net_resnet12_1xb105_mini-imagenet_5way-1shot.log.json) |
| [resnet12](/configs/classification/relation_net/mini_imagenet/relation-net_resnet12_1xb105_mini-imagenet_5way-5shot.py) |   84x84    |    105     |  5  |  5   |  71.31   | 0.37 |                                                                                       ⇑                                                                                       |                                                                            ⇑                                                                             |

## Results on Tiered-ImageNet dataset with 2000 episodes

| Arch                                                                                                                        | Input Size | Batch Size | way | shot | mean Acc | std  |                                                                                       ckpt                                                                                        |                                                                             log                                                                              |
| :-------------------------------------------------------------------------------------------------------------------------- | :--------: | :--------: | :-: | :--: | :------: | :--: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [conv4](/configs/classification/relation_net/tiered_imagenet/relation-net_conv4_1xb105_tiered-imagenet_5way-1shot.py)       |   84x84    |    105     |  5  |  1   |   47.6   | 0.48 |  [ckpt](https://download.openmmlab.com/mmfewshot/classification/relation_net/tiered_imagenet/relation-net_conv4_1xb105_tiered-imagenet_5way-1shot_20211120_141404-b41abf4d.pth)   |  [log](https://download.openmmlab.com/mmfewshot/classification/relation_net/tiered_imagenet/relation-net_conv4_1xb105_tiered-imagenet_5way-1shot.log.json)   |
| [conv4](/configs/classification/relation_net/tiered_imagenet/relation-net_conv4_1xb105_tiered-imagenet_5way-5shot.py)       |   84x84    |    105     |  5  |  5   |  64.09   | 0.43 |                                                                                         ⇑                                                                                         |                                                                              ⇑                                                                               |
| [resnet12](/configs/classification/relation_net/tiered_imagenet/relation-net_resnet12_1xb105_tiered-imagenet_5way-1shot.py) |   84x84    |    105     |  5  |  1   |  57.46   | 0.52 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/relation_net/tiered_imagenet/relation-net_resnet12_1xb105_tiered-imagenet_5way-1shot_20211120_145035-7df8c4d3.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/relation_net/tiered_imagenet/relation-net_resnet12_1xb105_tiered-imagenet_5way-1shot.log.json) |
| [resnet12](/configs/classification/relation_net/tiered_imagenet/relation-net_resnet12_1xb105_tiered-imagenet_5way-5shot.py) |   84x84    |    105     |  5  |  5   |  72.84   | 0.43 |                                                                                         ⇑                                                                                         |                                                                              ⇑                                                                               |
