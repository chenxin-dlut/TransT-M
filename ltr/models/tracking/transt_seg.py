import torch.nn as nn
import numpy as np

import torch
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2, interpolate,
                       accuracy)

from ltr.models.backbone.transt_backbone import build_backbone
from ltr.models.loss.matcher import build_matcher
from ltr.models.neck.featurefusion_network import build_featurefusion_network

class TransTsegm(nn.Module):
    """ This is the TransT module that performs single object tracking """
    def __init__(self, transt, freeze_transt=False):
        super().__init__()
        self.transt = transt

        if freeze_transt:
            for p in self.parameters():
                p.requires_grad_(False)

        hidden_dim, nheads = transt.featurefusion_network.d_model, transt.featurefusion_network.nhead
        self.bbox_attention_self = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0)
        self.bbox_attention_cross = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0)
        self.mask_head = MaskHeadSmallConv(hidden_dim + 2*nheads, [1024, 512, 256, 64], hidden_dim)


    def forward(self, search, template):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
        if not isinstance(template, NestedTensor):
            template = nested_tensor_from_tensor(template)
        feature_search, pos_search = self.transt.backbone(search)
        # feature_search: list of NestedTensor
        # feature_search[0].tensors.shape
        # torch.Size([16, 64, 128, 128])
        # feature_search[1].tensors.shape
        # torch.Size([16, 256, 64, 64])
        # feature_search[2].tensors.shape
        # torch.Size([16, 512, 32, 32])
        # feature_search[3].tensors.shape
        # torch.Size([16, 1024, 32, 32])
        feature_template, pos_template = self.transt.backbone(template)
        bs = feature_search[-1].tensors.shape[0]
        src_search, mask_search= feature_search[-1].decompose()
        # src_search: torch.Size([bs, 1024, 32, 32])
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None
        src_search_proj = self.transt.input_proj(src_search)
        src_template_proj = self.transt.input_proj(src_template)
        # src_search_proj.shape
        # torch.Size([16, 256, 32, 32])
        hs_search, hs_temp, memory = self.transt.featurefusion_network(src_template_proj, mask_template, src_search_proj, mask_search, pos_template[-1], pos_search[-1])
        # hs: torch.Size([1, bs, 1024, 256])
        outputs_class = self.transt.class_embed(hs_search)
        outputs_coord = self.transt.bbox_embed(hs_search).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        score = self._convert_score(outputs_class[-1]) # (16,1024)
        index_1 = score.argmax(axis=1)
        index_0 = np.arange(0, bs, 1)
        hs_search_choosen = hs_search[-1][index_0, index_1, :].unsqueeze(1)
        hs_temp_choosen = hs_temp[-1][:,119:120,:]

        #这里如果把hs[-1]共1024个向量全输入进取占显存太多，暂时取hs_temp中心的一个，并不是非常合理，后边修改
        bbox_mask_self = self.bbox_attention_self(hs_search_choosen, memory, mask=mask_search) #torch.Size([bs, n_q, nhead, h, w]) attention score 8-head 100-query
        bbox_mask_cross = self.bbox_attention_cross(hs_temp_choosen, memory, mask=mask_search) #torch.Size([bs, n_q, nhead, h, w]) attention score 8-head 100-query
        bbox_mask = torch.cat([bbox_mask_self, bbox_mask_cross], dim=2)

        #fpn的选择不一定合理
        seg_masks = self.mask_head(memory, bbox_mask, [feature_search[3].tensors, feature_search[2].tensors, feature_search[1].tensors, feature_search[0].tensors])
        num_queries_seg = 1
        outputs_seg_masks = seg_masks.view(bs, num_queries_seg, seg_masks.shape[-2], seg_masks.shape[-1])
        #
        out["pred_masks"] = outputs_seg_masks
        return out
        # out["pred_masks"].shape
        # torch.Size([16, 1, 64, 64])
        # out["pred_boxes"].shape
        # torch.Size([16, 1024, 4])
        # out["pred_logits"].shape
        # torch.Size([16, 1024, 2])

    def track(self, search, templates: list):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor_2(search)
        self.search_size = search.tensors.shape[-2:]
        self.feature_search, pos_search = self.transt.backbone(search)
        self.bs = self.feature_search[-1].tensors.shape[0]
        feature_template = self.transt.zf
        pos_template = self.transt.pos_template
        src_search, self.mask_search= self.feature_search[-1].decompose()
        assert self.mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None
        self.src_search_proj = self.transt.input_proj(src_search)
        src_template_proj = self.transt.input_proj(src_template)
        self.hs_search, self.hs_temp, self.memory = self.transt.featurefusion_network(src_template_proj, mask_template,
                                                                       self.src_search_proj, self.mask_search, pos_template[-1],
                                                                       pos_search[-1])
        outputs_class = self.transt.class_embed(self.hs_search)
        outputs_coord = self.transt.bbox_embed(self.hs_search).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    def track_seg(self, search, mask=True):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor_2(search)
        feature_search, pos_search = self.transt.backbone(search)
        bs = feature_search[-1].tensors.shape[0]
        feature_template = self.transt.zf
        pos_template = self.transt.pos_template
        src_search, mask_search= feature_search[-1].decompose()
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None
        src_search_proj = self.transt.input_proj(src_search)
        src_template_proj = self.transt.input_proj(src_template)
        hs_search, hs_temp, memory = self.transt.featurefusion_network(src_template_proj, mask_template,
                                                                       src_search_proj, mask_search, pos_template[-1],
                                                                       pos_search[-1])
        outputs_class = self.transt.class_embed(hs_search)
        outputs_coord = self.transt.bbox_embed(hs_search).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if mask==True:
            score = self._convert_score(outputs_class[-1])  # (16,1024)

            index_1 = score.argmax(axis=1)
            index_0 = np.arange(0, bs, 1)
            hs_search_choosen = hs_search[-1][index_0, index_1, :].unsqueeze(1)

            hs_temp_choosen = hs_temp[-1][:, 119:120, :]

            bbox_mask_self = self.bbox_attention_self(hs_search_choosen, memory,
                                                      mask=mask_search)  # torch.Size([bs, n_q, nhead, h, w]) attention score 8-head 100-query
            bbox_mask_cross = self.bbox_attention_cross(hs_temp_choosen, memory,
                                                        mask=mask_search)  # torch.Size([bs, n_q, nhead, h, w]) attention score 8-head 100-query
            bbox_mask = torch.cat([bbox_mask_self, bbox_mask_cross], dim=2)

            # fpn的选择不一定合理
            seg_masks = self.mask_head(memory, bbox_mask,
                                       [feature_search[3].tensors, feature_search[2].tensors,
                                        feature_search[1].tensors, feature_search[0].tensors])
            num_queries_seg = 1
            outputs_seg_masks = seg_masks.view(bs, num_queries_seg, seg_masks.shape[-2], seg_masks.shape[-1])
            #
            outputs_seg_masks = interpolate(outputs_seg_masks, size=search.tensors.shape[-2:],
                                    mode="bilinear", align_corners=False)
            out["pred_masks"] = outputs_seg_masks.sigmoid()
        return out

    def template(self, z):
        if not isinstance(z, NestedTensor):
            z = nested_tensor_from_tensor_2(z)
        zf, pos_template = self.transt.backbone(z)
        self.transt.zf = zf
        self.transt.pos_template = pos_template

    def _convert_score(self, score):
        score = F.softmax(score, dim=2).data[:, :, 0].cpu().numpy()
        return score
        #score.shape (16, 1024)


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 32] # [264, 128, 64, 32, 16, 8]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1) # 264->264
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)  # 264->128
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)   # 128->64
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)   # 64->32
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)   # 32->16
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.lay6 = torch.nn.Conv2d(inter_dims[4], inter_dims[5], 3, padding=1)   # 16->8
        self.gn6 = torch.nn.GroupNorm(8, inter_dims[5])
        self.out_lay = torch.nn.Conv2d(inter_dims[5], 1, 3, padding=1)   # 8->1

        self.dim = dim #264

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)   # 1024->128
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)   # 512->64
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)   # 256->32
        self.adapter4 = torch.nn.Conv2d(fpn_dims[3], inter_dims[4], 1)   # 64->16

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, bbox_mask, fpns):
        # x torch.Size([16, 256, 32, 32])
        # bbox_mask torch.Size([16, 1024, 8, 32, 32])
        # fpns[0] torch.Size([16, 1024, 32, 32])
        # fpns[1] torch.Size([16, 512, 32, 32])
        # fpns[2] torch.Size([16, 256, 64, 64])

        def expand(tensor, length):
            return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)

        x = torch.cat([expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)
        # 把attention score和主干网络的特征cat到一起
        # expand(x, bbox_mask.shape[1]).shape
        # torch.Size([bs*1024, 256, 32, 232])
        # bbox_mask.flatten(0, 1).shape
        # torch.Size([bs*1024, 8, 32, 32])
        # x.shape
        # torch.Size([bs*1024, 264, 32, 32])

        x = self.lay1(x) #torch.Size([bs*1024, 264, 32, 32])
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x) #torch.Size([bs*1024, 128, 32, 32])
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0]) #torch.Size([bs, 1024, 32, 32])-> torch.Size([bs, 128, 32, 32])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0)) #torch.Size([bs*1024, 128, 32, 32])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest") #与浅层的特征相加
        x = self.lay3(x)   # 128->64
        x = self.gn3(x)
        x = F.relu(x) #torch.Size([bs*1024, 64, 32, 32])

        cur_fpn = self.adapter2(fpns[1]) #torch.Size([bs, 512, 32, 32]) -> torch.Size([bs, 64, 32, 32])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0)) #torch.Size([bs*1024, 64, 32, 32])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)   # 64->32
        x = self.gn4(x)
        x = F.relu(x) #torch.Size([bs*1024, 32, 32, 32])

        cur_fpn = self.adapter3(fpns[2]) #torch.Size([16, 256, 64, 64]) -> torch.Size([16, 32, 64, 64])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0)) #torch.Size([bs*1024, 32, 64, 64])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest") #torch.Size([bs*1024, 32, 64, 64])
        x = self.lay5(x)   # 32->16
        x = self.gn5(x)
        x = F.relu(x)

        cur_fpn = self.adapter4(fpns[3]) #torch.Size([16, 64, 128, 128]) -> torch.Size([16, 16, 128, 128])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0)) #torch.Size([bs*1024, 16, 128, 128])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest") #torch.Size([bs*1024, 16, 128, 128])
        x = self.lay6(x)   # 16->8
        x = self.gn6(x)
        x = F.relu(x)

        x = self.out_lay(x)   # 8->1 #torch.Size([bs*1024, 1, 128, 128])
        return x


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask=None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes



