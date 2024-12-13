import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..registry import NECKS
@NECKS.register_module


class Aggregator(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 attention=False,
                 cfg = None):
        super(Aggregator, self).__init__()
        assert isinstance(in_channels, list) and len(in_channels) == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_cfg = upsample_cfg.copy()
        self.attention = attention

        # Initialize lateral and Aggregator convolution layers
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(len(self.in_channels)):
            l_conv = ConvModule(
                self.in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False
            )

            f_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(f_conv)

    def forward(self, inputs):
        input_tensor = inputs[-1]

        laterals = [self.lateral_convs[i](input_tensor) for i in range(len(self.in_channels))]
        output = self.fpn_convs[0](laterals[0])

        return (output,)

