# Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks <a href="https://arxiv.org/abs/1703.03400"> (ICML'2017)</a>

## Abstract

<!-- [ABSTRACT] -->

We propose an algorithm for meta-learning that
is model-agnostic, in the sense that it is compatible
with any model trained with gradient descent
and applicable to a variety of different
learning problems, including classification, regression,
and reinforcement learning. The goal
of meta-learning is to train a model on a variety
of learning tasks, such that it can solve new
learning tasks using only a small number of training
samples. In our approach, the parameters of
the model are explicitly trained such that a small
number of gradient steps with a small amount
of training data from a new task will produce
good generalization performance on that task. In
effect, our method trains the model to be easy
to fine-tune. We demonstrate that this approach
leads to state-of-the-art performance on two fewshot
image classification benchmarks, produces
good results on few-shot regression, and accelerates
fine-tuning for policy gradient reinforcement
learning with neural network policies.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/15669896/142851409-8dd06edd-2204-4106-9a1b-e6b318ea80cc.png" width="50%"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{FinnAL17,
  title={Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks},
  author={Chelsea Finn and Pieter Abbeel and Sergey Levine},
  booktitle={Proceedings of the 34th International Conference on Machine Learning},
  year={2017}
}
```

## How to Reproduce MAML

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
  configs/classification/maml/cub/maml_conv4_1xb105_cub_5way-1shot.py

# meta testing
python ./tools/classification/test.py \
  configs/classification/maml/cub/maml_conv4_1xb105_cub_5way-1shot.py \
  work_dir/maml_conv4_1xb105_cub_5way-1shot/best_accuracy_mean.pth
```

**Note**:

- All the result are trained with single gpu.
- The configs of 1 shot and 5 shot use same training setting,
  but different meta test setting on validation set and test set.
- The hyper-parameters in configs are roughly set and probably not the optimal one so
  feel free to tone and try different configurations.
  For example, try different learning rate or validation episodes for each setting.
  Anyway, we will continue to improve it.

## Results on CUB dataset with 2000 episodes

| Arch                                                                                | Input Size | Batch Size | way | shot | mean Acc | std  |                                                                   ckpt                                                                    |                                                         log                                                          |
| :---------------------------------------------------------------------------------- | :--------: | :--------: | :-: | :--: | :------: | :--: | :---------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: |
| [conv4](/configs/classification/maml/cub/maml_conv4_1xb105_cub_5way-1shot.py)       |   84x84    |    105     |  5  |  1   |  60.32   | 0.5  |  [ckpt](https://download.openmmlab.com/mmfewshot/classification/maml/cub/maml_conv4_1xb105_cub_5way-1shot_20211031_125617-c00d532b.pth)   |  [log](https://download.openmmlab.com/mmfewshot/classification/maml/cub/maml_conv4_1xb105_cub_5way-1shot.log.json)   |
| [conv4](/configs/classification/maml/cub/maml_conv4_1xb105_cub_5way-5shot.py)       |   84x84    |    105     |  5  |  5   |  77.03   | 0.39 |  [ckpt](https://download.openmmlab.com/mmfewshot/classification/maml/cub/maml_conv4_1xb105_cub_5way-5shot_20211031_125919-48d15b7a.pth)   |  [log](https://download.openmmlab.com/mmfewshot/classification/maml/cub/maml_conv4_1xb105_cub_5way-5shot.log.json)   |
| [resnet12](/configs/classification/maml/cub/maml_resnet12_1xb105_cub_5way-1shot.py) |   84x84    |    105     |  5  |  1   |  70.44   | 0.55 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/maml/cub/maml_resnet12_1xb105_cub_5way-1shot_20211031_130240-542c4387.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/maml/cub/maml_resnet12_1xb105_cub_5way-1shot.log.json) |
| [resnet12](/configs/classification/maml/cub/maml_resnet12_1xb105_cub_5way-5shot.py) |   84x84    |    105     |  5  |  5   |   85.5   | 0.33 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/maml/cub/maml_resnet12_1xb105_cub_5way-5shot_20211031_130534-1de7c6d3.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/maml/cub/maml_resnet12_1xb105_cub_5way-5shot.log.json) |

## Results on Mini-ImageNet dataset with 2000 episodes

| Arch                                                                                                    | Input Size | Batch Size | way | shot | mean Acc | std  |                                                                             ckpt                                                                              |                                                                   log                                                                    |
| :------------------------------------------------------------------------------------------------------ | :--------: | :--------: | :-: | :--: | :------: | :--: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------: |
| [conv4](/configs/classification/maml/mini_imagenet/maml_conv4_1xb105_mini-imagenet_5way-1shot.py)       |   84x84    |    105     |  5  |  1   |  46.76   | 0.42 |  [ckpt](https://download.openmmlab.com/mmfewshot/classification/maml/mini_imagenet/maml_conv4_1xb105_mini-imagenet_5way-1shot_20211104_072004-02e6c7a7.pth)   |  [log](https://download.openmmlab.com/mmfewshot/classification/maml/mini_imagenet/maml_conv4_1xb105_mini-imagenet_5way-1shot.log.json)   |
| [conv4](/configs/classification/maml/mini_imagenet/maml_conv4_1xb105_mini-imagenet_5way-5shot.py)       |   84x84    |    105     |  5  |  5   |  63.88   | 0.39 |  [ckpt](https://download.openmmlab.com/mmfewshot/classification/maml/mini_imagenet/maml_conv4_1xb105_mini-imagenet_5way-5shot_20211104_072004-4c336eec.pth)   |  [log](https://download.openmmlab.com/mmfewshot/classification/maml/mini_imagenet/maml_conv4_1xb105_mini-imagenet_5way-5shot.log.json)   |
| [resnet12](/configs/classification/maml/mini_imagenet/maml_resnet12_1xb105_mini-imagenet_5way-1shot.py) |   84x84    |    105     |  5  |  1   |   57.4   | 0.47 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/maml/mini_imagenet/maml_resnet12_1xb105_mini-imagenet_5way-1shot_20211104_105317-d1628e14.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/maml/mini_imagenet/maml_resnet12_1xb105_mini-imagenet_5way-1shot.log.json) |
| [resnet12](/configs/classification/maml/mini_imagenet/maml_resnet12_1xb105_mini-imagenet_5way-5shot.py) |   84x84    |    105     |  5  |  5   |  72.42   | 0.38 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/maml/mini_imagenet/maml_resnet12_1xb105_mini-imagenet_5way-5shot_20211104_105516-13547c7b.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/maml/mini_imagenet/maml_resnet12_1xb105_mini-imagenet_5way-5shot.log.json) |

## Results on Tiered-ImageNet dataset with 2000 episodes

| Arch                                                                                                        | Input Size | Batch Size | way | shot | mean Acc | std  |                                                                               ckpt                                                                                |                                                                     log                                                                      |
| :---------------------------------------------------------------------------------------------------------- | :--------: | :--------: | :-: | :--: | :------: | :--: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------: |
| [conv4](/configs/classification/maml/tiered_imagenet/maml_conv4_1xb105_tiered-imagenet_5way-1shot.py)       |   84x84    |    105     |  5  |  1   |  45.56   | 0.49 |  [ckpt](https://download.openmmlab.com/mmfewshot/classification/maml/tiered_imagenet/maml_conv4_1xb105_tiered-imagenet_5way-1shot_20211104_110055-a08f6d09.pth)   |  [log](https://download.openmmlab.com/mmfewshot/classification/maml/tiered_imagenet/maml_conv4_1xb105_tiered-imagenet_5way-1shot.log.json)   |
| [conv4](/configs/classification/maml/tiered_imagenet/maml_conv4_1xb105_tiered-imagenet_5way-5shot.py)       |   84x84    |    105     |  5  |  5   |   60.2   | 0.43 |  [ckpt](https://download.openmmlab.com/mmfewshot/classification/maml/tiered_imagenet/maml_conv4_1xb105_tiered-imagenet_5way-5shot_20211104_110734-88a559bc.pth)   |  [log](https://download.openmmlab.com/mmfewshot/classification/maml/tiered_imagenet/maml_conv4_1xb105_tiered-imagenet_5way-5shot.log.json)   |
| [resnet12](/configs/classification/maml/tiered_imagenet/maml_resnet12_1xb105_tiered-imagenet_5way-1shot.py) |   84x84    |    105     |  5  |  1   |  57.63   | 0.53 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/maml/tiered_imagenet/maml_resnet12_1xb105_tiered-imagenet_5way-1shot_20211104_115442-340cb23f.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/maml/tiered_imagenet/maml_resnet12_1xb105_tiered-imagenet_5way-1shot.log.json) |
| [resnet12](/configs/classification/maml/tiered_imagenet/maml_resnet12_1xb105_tiered-imagenet_5way-5shot.py) |   84x84    |    105     |  5  |  5   |   72.3   | 0.43 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/maml/tiered_imagenet/maml_resnet12_1xb105_tiered-imagenet_5way-5shot_20211104_115442-95920719.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/maml/tiered_imagenet/maml_resnet12_1xb105_tiered-imagenet_5way-5shot.log.json) |
