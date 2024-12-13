import torch.nn as nn
import torch

from clrkd.models.registry import NETS
from ..registry import build_backbones, build_heads, build_necks


@NETS.register_module
class Detector(nn.Module):
    def __init__(self, cfg):
        super(Detector, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbones(cfg)
        self.neck = build_necks(cfg) if cfg.haskey('neck') else None
        self.heads = build_heads(cfg)
    
    def get_lanes(self):
        return self.heads.get_lanes(output)

    def forward(self, batch, attention = False):
        output = {}

        if attention:
            fea, att_maps = self.backbone(batch['img'] if isinstance(batch, dict) else batch, attention)
        else:
            fea = self.backbone(batch['img'] if isinstance(batch, dict) else batch)

        if self.neck:
            fea = self.neck(fea)


        # import pdb; pdb.set_trace()
        if attention:
            output, logits_att, priors_att = self.heads(fea, attention, batch=batch)
        elif self.training:
            output = self.heads(fea, batch=batch)
        else:
            output = self.heads(fea)

        if attention:
            return output, att_maps, logits_att, priors_att
        else:
            return output