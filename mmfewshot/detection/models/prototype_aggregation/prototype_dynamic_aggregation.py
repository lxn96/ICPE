import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from ..builder import PROTOTYPE_AGGREGATION


@PROTOTYPE_AGGREGATION.register_module()
class PrototypeDynamicAggregation(BaseModule):

    def __init__(self,
                 in_channels=2048,
                 alpha=1.,
                 init_cfg=None
                 ):
        super(PrototypeDynamicAggregation, self).__init__(init_cfg)
        self.fc_cls = nn.Linear(in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=3)
        self.alpha = alpha

    def forward(self, x, label):
        x_mean = x.mean(3).mean(2)
        x_mean_1 = x_mean.view(x_mean.size(0), -1, 1, 1).expand_as(x)
        sim = torch.cosine_similarity(x, x_mean_1, dim=1).unsqueeze(1)
        x_new = (x * sim.expand_as(x)).mean(3).mean(2)
        x = self.alpha * x_new + x_mean

        weights = self.fc_cls(x)
        label = torch.cat(label)
        out_feats = []
        out_labels = []
        labels = torch.unique(label, sorted=False)
        labels = labels[torch.randperm(labels.size(0))]
        for cls_id in labels:
            out_feat = x[label == cls_id]
            weight = self.sigmoid(weights[label == cls_id])
            out_feat = torch.mean(out_feat * weight.expand_as(out_feat), dim=0).unsqueeze(0)

            out_feats.append(out_feat)
            out_labels.append(torch.tensor([cls_id]).type_as(label))

        return torch.cat(out_feats, dim=0), out_labels

    def simple_test(self, x):
        x_mean = x.mean(3).mean(2)
        x_mean_1 = x_mean.view(x_mean.size(0), -1, 1, 1).expand_as(x)
        sim = torch.cosine_similarity(x, x_mean_1, dim=1).unsqueeze(1)
        x_new = (x * sim.expand_as(x)).mean(3).mean(2)
        x = self.alpha * x_new + x_mean

        weight = self.fc_cls(x)
        weight = self.sigmoid(weight)
        out_feat = torch.mean(x * weight.expand_as(x), dim=0).unsqueeze(0)
        return out_feat


