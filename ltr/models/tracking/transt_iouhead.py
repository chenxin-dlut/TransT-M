import torch.nn as nn
import numpy as np

import torch
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2, interpolate,
                       nested_tensor_from_tensor_list, accuracy)

from ltr.models.backbone.transt_backbone import build_backbone
from ltr.models.loss.matcher import build_matcher
from ltr.models.neck.featurefusion_network import build_featurefusion_network


class TransTiouh(nn.Module):
    """ This is the TransT module that performs single object tracking """
    def __init__(self, transt, freeze_transt=False):
        super().__init__()
        self.featurefusion_network = transt.featurefusion_network
        self.class_embed = transt.class_embed
        self.bbox_embed = transt.bbox_embed
        self.input_proj = transt.input_proj
        self.backbone = transt.backbone

        if freeze_transt:
            for p in self.parameters():
                p.requires_grad_(False)

        hidden_dim = self.featurefusion_network.d_model
        self.iou_embed = MLP(hidden_dim + 4, hidden_dim, 1, 3)

    def forward(self, search, templates):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
        feature_search, pos_search = self.backbone(search)
        src_search, mask_search= feature_search[-1].decompose()
        src_search = self.input_proj(src_search)

        bs, n_t, c, h, w = templates.shape
        templates = templates.reshape(bs * n_t, c, h, w)
        if not isinstance(templates, NestedTensor):
            templates = nested_tensor_from_tensor(templates)
        feature_templates, pos_templates = self.backbone(templates)
        src_templates, mask_templates = feature_templates[-1].decompose()
        src_templates = self.input_proj(src_templates)
        _, c_src, h_src, w_src = src_templates.shape
        pos_templates = pos_templates[-1].reshape(bs, n_t, c_src, h_src, w_src)
        src_templates = src_templates.reshape(bs, n_t, c_src, h_src, w_src)
        mask_templates = mask_templates.reshape(bs, n_t, h_src, w_src)

        hs, _, _ = self.featurefusion_network(src_templates, mask_templates, pos_templates,
                                              src_search, mask_search, pos_search[-1])

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        outputs_iouh = self.iou_embed(torch.cat((hs, outputs_coord), 3)).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_iouh': outputs_iouh[-1]}
        return out

    def track(self, search, templates: list):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor_2(search)
        feature_search, pos_search = self.backbone(search)
        src_search, mask_search= feature_search[-1].decompose()
        src_search = self.input_proj(src_search)

        for i in range(len(templates)):
            if i == 0:
                src_templates = templates[i]['src']
                mask_templates = templates[i]['mask']
                pos_templates = templates[i]['pos']
            else:
                src_templates = torch.cat((src_templates, templates[i]['src']), 1)
                mask_templates = torch.cat((mask_templates, templates[i]['mask']), 1)
                pos_templates = torch.cat((pos_templates, templates[i]['pos']), 1)

        hs, _, _ = self.featurefusion_network(src_templates, mask_templates, pos_templates,
                                              src_search, mask_search, pos_search[-1])

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        outputs_iouh = self.iou_embed(torch.cat((hs, outputs_coord), 3)).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_iouh': outputs_iouh[-1]}
        return out

    def template(self, z):
        if not isinstance(z, NestedTensor):
            z = nested_tensor_from_tensor_2(z)
        feature_template, pos_template = self.backbone(z)
        src_template, mask_template = feature_template[-1].decompose()
        template_out = {
            'pos': pos_template[-1].unsqueeze(1),
            'src': self.input_proj(src_template).unsqueeze(1),
            'mask': mask_template.unsqueeze(1)
        }
        return template_out

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
