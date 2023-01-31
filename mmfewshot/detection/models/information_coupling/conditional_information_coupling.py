import torch
from torch import nn
from torch.nn import functional as F
from mmcv.runner import BaseModule
from ..builder import INFORMATION_COUPLING


@INFORMATION_COUPLING.register_module()
class ConditionalInformationCouplingModule(BaseModule):

    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=2,
                 sub_sample=True,
                 bn_layer=True,
                 init_cfg=None):
        super(ConditionalInformationCouplingModule, self).__init__(init_cfg)

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.v = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.q = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.k = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        if sub_sample:
            self.v = nn.Sequential(self.v, max_pool_layer)
            self.k = nn.Sequential(self.k, max_pool_layer)

    def forward(self, x, kv_x):
        return self.forward_single(x, kv_x)

    def forward_single(self, x, kv_x):

        batch_size = x.size(0)

        v_x = self.v(kv_x).view(batch_size, self.inter_channels, -1)
        v_x = v_x.permute(0, 2, 1)

        q_x = self.q(x).view(batch_size, self.inter_channels, -1)
        q_x = q_x.permute(0, 2, 1)

        k_x = self.k(kv_x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(q_x, k_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, v_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)

        kv_x = (self.gap(kv_x).view(kv_x.size(0), -1, 1, 1)).expand_as(x)
        mask = torch.cosine_similarity(x, kv_x, dim=1).unsqueeze(1)
        z = W_y * mask.expand_as(W_y) + x

        return z
