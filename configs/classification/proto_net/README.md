# Prototypical Networks for Few-shot Learning <a href="https://arxiv.org/abs/1703.05175"> (NeurIPS'2017)</a>

<!-- [ABSTRACT] -->

We propose prototypical networks for the problem of few-shot classification, where
a classifier must generalize to new classes not seen in the training set, given only
a small number of examples of each new class. Prototypical networks learn a
metric space in which classification can be performed by computing distances
to prototype representations of each class. Compared to recent approaches for
few-shot learning, they reflect a simpler inductive bias that is beneficial in this
limited-data regime, and achieve excellent results. We provide an analysis showing
that some simple design decisions can yield substantial improvements over recent
approaches involving complicated architectural choices and meta-learning. We
further extend prototypical networks to zero-shot learning and achieve state-of-the-art
results on the CU-Birds dataset.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/15669896/142848746-475a9e0d-90bd-42f9-965c-dd514c65f05b.png" width="80%"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{snell2017prototypical,
    title={Prototypical networks for few-shot learning},
    author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
    booktitle={Proceedings of the 31st International Conference on Neural Information Processing Systems},
    pages={4080--4090},
    year={2017}
}
```

## How to Reproduce ProtoNet

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
  configs/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-1shot.py

# meta testing
python ./tools/classification/test.py \
  configs/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-1shot.py \
  work_dir/proto-net_conv4_1xb105_cub_5way-1shot/best_accuracy_mean.pth
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

| Arch                                                                                          | Input Size | Batch Size | way | shot | mean Acc | std  |                                                                        ckpt                                                                         |                                                              log                                                               |
| :-------------------------------------------------------------------------------------------- | :--------: | :--------: | :-: | :--: | :------: | :--: | :-------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
| [conv4](/configs/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-1shot.py)       |   84x84    |    105     |  5  |  1   |  58.86   | 0.52 |  [ckpt](https://download.openmmlab.com/mmfewshot/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-1shot_20211120_101211-9ab530c3.pth)   |  [log](https://download.openmmlab.com/mmfewshot/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-1shot.log.json)   |
| [conv4](/configs/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-5shot.py)       |   84x84    |    105     |  5  |  5   |  80.77   | 0.34 |                                                                          ⇑                                                                          |                                                               ⇑                                                                |
| [resnet12](/configs/classification/proto_net/cub/proto-net_resnet12_1xb105_cub_5way-1shot.py) |   84x84    |    105     |  5  |  1   |  74.35   | 0.48 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/proto_net/cub/proto-net_resnet12_1xb105_cub_5way-1shot_20211120_101211-da5bfb99.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/proto_net/cub/proto-net_resnet12_1xb105_cub_5way-1shot.log.json) |
| [resnet12](/configs/classification/proto_net/cub/proto-net_resnet12_1xb105_cub_5way-5shot.py) |   84x84    |    105     |  5  |  5   |   88.5   | 0.25 |                                                                          ⇑                                                                          |                                                               ⇑                                                                |

## Results on Mini-ImageNet dataset with 2000 episodes

| Arch                                                                                                              | Input Size | Batch Size | way | shot | mean Acc | std  |                                                                                  ckpt                                                                                   |                                                                        log                                                                         |
| :---------------------------------------------------------------------------------------------------------------- | :--------: | :--------: | :-: | :--: | :------: | :--: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------: |
| [conv4](/configs/classification/proto_net/mini_imagenet/proto-net_conv4_1xb105_mini-imagenet_5way-1shot.py)       |   84x84    |    105     |  5  |  1   |  48.11   | 0.43 |  [ckpt](https://download.openmmlab.com/mmfewshot/classification/proto_net/mini_imagenet/proto-net_conv4_1xb105_mini-imagenet_5way-1shot_20211120_134319-646809cf.pth)   |  [log](https://download.openmmlab.com/mmfewshot/classification/proto_net/mini_imagenet/proto-net_conv4_1xb105_mini-imagenet_5way-1shot.log.json)   |
| [conv4](/configs/classification/proto_net/mini_imagenet/proto-net_conv4_1xb105_mini-imagenet_5way-5shot.py)       |   84x84    |    105     |  5  |  5   |  68.51   | 0.37 |                                                                                    ⇑                                                                                    |                                                                         ⇑                                                                          |
| [resnet12](/configs/classification/proto_net/mini_imagenet/proto-net_resnet12_1xb105_mini-imagenet_5way-1shot.py) |   84x84    |    105     |  5  |  1   |  56.13   | 0.45 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/proto_net/mini_imagenet/proto-net_resnet12_1xb105_mini-imagenet_5way-1shot_20211120_134319-73173bee.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/proto_net/mini_imagenet/proto-net_resnet12_1xb105_mini-imagenet_5way-1shot.log.json) |
| [resnet12](/configs/classification/proto_net/mini_imagenet/proto-net_resnet12_1xb105_mini-imagenet_5way-5shot.py) |   84x84    |    105     |  5  |  5   |   75.7   | 0.33 |                                                                                    ⇑                                                                                    |                                                                         ⇑                                                                          |

## Results on Tiered-ImageNet dataset with 2000 episodes

| Arch                                                                                                                  | Input Size | Batch Size | way | shot | mean Acc | std  |                                                                                    ckpt                                                                                     |                                                                          log                                                                           |
| :-------------------------------------------------------------------------------------------------------------------- | :--------: | :--------: | :-: | :--: | :------: | :--: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
| [conv4](/configs/classification/proto_net/tiered_imagenet/proto-net_conv4_1xb105_tiered-imagenet_5way-1shot.py)       |   84x84    |    105     |  5  |  1   |   45.5   | 0.46 |  [ckpt](https://download.openmmlab.com/mmfewshot/classification/proto_net/tiered_imagenet/proto-net_conv4_1xb105_tiered-imagenet_5way-1shot_20211120_134742-26520ca8.pth)   |  [log](https://download.openmmlab.com/mmfewshot/classification/proto_net/tiered_imagenet/proto-net_conv4_1xb105_tiered-imagenet_5way-1shot.log.json)   |
| [conv4](/configs/classification/proto_net/tiered_imagenet/proto-net_conv4_1xb105_tiered-imagenet_5way-5shot.py)       |   84x84    |    105     |  5  |  5   |  62.89   | 0.43 |                                                                                      ⇑                                                                                      |                                                                           ⇑                                                                            |
| [resnet12](/configs/classification/proto_net/tiered_imagenet/proto-net_resnet12_1xb105_tiered-imagenet_5way-1shot.py) |   84x84    |    105     |  5  |  1   |  59.11   | 0.52 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/proto_net/tiered_imagenet/proto-net_resnet12_1xb105_tiered-imagenet_5way-1shot_20211120_153230-eb72884e.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/proto_net/tiered_imagenet/proto-net_resnet12_1xb105_tiered-imagenet_5way-1shot.log.json) |
| [resnet12](/configs/classification/proto_net/tiered_imagenet/proto-net_resnet12_1xb105_tiered-imagenet_5way-5shot.py) |   84x84    |    105     |  5  |  5   |   75.3   | 0.42 |                                                                                      ⇑                                                                                      |                                                                           ⇑                                                                            |
