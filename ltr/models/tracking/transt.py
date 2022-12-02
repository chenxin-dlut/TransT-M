import torch.nn as nn
from ltr import model_constructor

import torch
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2, interpolate,
                       nested_tensor_from_tensor_list, accuracy)

from ltr.models.backbone.transt_backbone import build_backbone
from ltr.models.loss.matcher import build_matcher
from ltr.models.neck.featurefusion_network import build_featurefusion_network
from ltr.models.tracking.transt_seg import (TransTsegm, dice_loss, sigmoid_focal_loss)
from ltr.models.tracking.transt_iouhead import TransTiouh
from ltr.models.tracking.transt_iouh_seg import TransTiouhsegm


class TransT(nn.Module):
    """ This is the TransT module that performs single object tracking """
    def __init__(self, backbone, featurefusion_network, num_classes):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See transt_backbone.py
            featurefusion_network: torch module of the featurefusion_network architecture, a variant of transformer.
                                   See featurefusion_network.py
            num_classes: number of object classes, always 1 for single object tracking
        """
        super().__init__()
        self.featurefusion_network = featurefusion_network
        hidden_dim = featurefusion_network.d_model
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, search, templates):
        """ The forward expects a NestedTensor, which consists of:
               - search.tensors: batched images, of shape [batch_size x 3 x H_search x W_search]
               - search.mask: a binary mask of shape [batch_size x H_search x W_search], containing 1 on padded pixels
               - template.tensors: batched images, of shape [batch_size x 3 x H_template x W_template]
               - template.mask: a binary mask of shape [batch_size x H_template x W_template], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits for all feature vectors.
                                Shape= [batch_size x num_vectors x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all feature vectors, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image.

        """
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
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
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
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
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

class SetCriterion(nn.Module):
    """ This class computes the loss for TransT.
    The process happens in two steps:
        1) we compute assignment between ground truth box and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, always be 1 for single object tracking.
            matcher: module able to compute a matching between target and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        # self.iouhead_loss = nn.MSELoss()

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        giou, iou = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))
        giou = torch.diag(giou)
        iou = torch.diag(iou)
        loss_giou = 1 - giou
        iou = iou
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        losses['iou'] = iou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_masks = outputs["pred_masks"] # torch.Size([bs, 1, 128, 128])

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose() #torch.Size([32, 1, 256, 256])
        target_masks = target_masks.to(src_masks) #torch.Size([bs, 1, 256, 256])

        # upsample predictions to the target size
        src_masks = interpolate(src_masks, size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        #torch.Size([bs, 1, 256, 256])

        src_masks = src_masks[:, 0].flatten(1) #torch.Size([18, 660969])

        target_masks = target_masks[:, 0].flatten(1) #torch.Size([18, 660969])

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, src_masks.shape[0]),
            "loss_dice": dice_loss(src_masks, target_masks, src_masks.shape[0]),
        }
        return losses

    def loss_iouh(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_iouh" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_iouh = outputs['pred_iouh'][idx]
        with torch.no_grad():
            src_boxes = outputs['pred_boxes'][idx]
            # target_boxes = torch.cat([torch.cat([target['boxes']] * 1024, 0) for target in targets], 0)
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            giou, iou = box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
            iou = iou.unsqueeze(1)

        # src_iouh.shape
        # torch.Size([3799, 1])
        # iou.shape
        # torch.Size([3799, 1])

        losses = {
            "loss_iouh": self.iouhead_loss(src_iouh, iou),
        }
        return losses

    def iouhead_loss(self, src_iouh, iou):
        loss = torch.mean(((1-iou)**2)*((src_iouh - iou)**2))
        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'iouh': self.loss_iouh
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the target
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_pos = sum(len(t[0]) for t in indices)

        num_boxes_pos = torch.as_tensor([num_boxes_pos], dtype=torch.float, device=next(iter(outputs.values())).device)

        num_boxes_pos = torch.clamp(num_boxes_pos, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes_pos))

        return losses

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


@model_constructor
def transt_resnet50(settings):
    num_classes = 1
    backbone_net = build_backbone(settings, backbone_pretrained=True)
    featurefusion_network = build_featurefusion_network(settings)
    model = TransT(
        backbone_net,
        featurefusion_network,
        num_classes=num_classes
    )
    if settings.iou_head:
        assert settings.masks == False
        model = TransTiouh(model, freeze_transt=settings.freeze_transt)
    elif settings.masks:
        assert settings.iou_head == False
        model = TransTiouh(model, freeze_transt=settings.freeze_transt)
        model = TransTiouhsegm(model, freeze_transt=settings.freeze_transt)
    device = torch.device(settings.device)
    model.to(device)
    return model

def transt_loss(settings):
    num_classes = 1
    matcher = build_matcher()
    weight_dict = {'loss_ce': 8.334, 'loss_bbox': 5}
    weight_dict['loss_giou'] = 2
    if settings.masks:
        weight_dict["loss_mask"] = 1
        weight_dict["loss_dice"] = 1
    if settings.iou_head:
        weight_dict["loss_iouh"] = 1
    losses = ['labels', 'boxes']
    if settings.masks:
        losses += ["masks"]
    if settings.iou_head:
        losses += ["iouh"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.0625, losses=losses)
    device = torch.device(settings.device)
    criterion.to(device)
    return criterion
