import torch
import numpy as np
from pytracking import TensorDict
from ltr.admin import multigpu
import cv2

class TranstActor:
    """ Actor for training the TransT"""
    def __init__(self, net, objective, settings):
        """
        args:
            net - The network to train
            objective - The loss function
        """
        self.net = net
        self.objective = objective
        self.settings = settings

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the fields 'search_images', 'template_images', 'search_anno'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # a = data['search_images'][0].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('a',a)
        # b = data['static_template_images'][0].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('b',b)
        # c = data['dynamic_template_images'][0,0].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('c',c)
        # d = data['dynamic_template_images'][0,1].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('d',d)
        # cv2.waitKey(0)
        # b = data['search_anno'][0].cpu().numpy()
        # cv2.rectangle(a, (int(b[0]), int(b[1])), (int(b[0]+b[2]), int(b[1]+b[3])), (0, 1.0, 0), 2)
        outputs = self.net(data['search_images'], data['template_images'])
        # generate labels
        # data['search_masks'].shape: torch.Size([bs, 1, 256, 256])

        targets =[]
        targets_origin = data['search_anno']
        _, _, h, w = data['search_images'].shape
        targets_origin[:, 0] += targets_origin[:, 2] / 2
        targets_origin[:, 0] /= w
        targets_origin[:, 1] += targets_origin[:, 3] / 2
        targets_origin[:, 1] /= h
        targets_origin[:, 2] /= w
        targets_origin[:, 3] /= h
        targets_origin = targets_origin.unsqueeze(1)
        for i in range(len(targets_origin)):
            target_origin = targets_origin[i]
            target = {}
            target['boxes'] = target_origin
            label = np.array([0])
            label = torch.tensor(label, device=data['search_anno'].device)
            target['labels'] = label
            target['masks'] = data['search_masks'][i]
            targets.append(target)


        # Compute loss
        # outputs:(center_x, center_y, width, height)
        loss_dict = self.objective(outputs, targets)
        weight_dict = self.objective.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Return training stats
        stats = {'Loss/total': losses.item(),
                 'Loss/ce': loss_dict['loss_ce'].item(),
                 'Loss/bbox': loss_dict['loss_bbox'].item(),
                 'Loss/giou': loss_dict['loss_giou'].item(),
                 'iou': loss_dict['iou'].item()
                 }
        if self.settings.masks:
            stats['Loss/mask'] = loss_dict['loss_mask'].item()
            stats['Loss/dice'] = loss_dict['loss_dice'].item()
        if self.settings.iou_head:
            stats['Loss/iou_head'] = loss_dict['loss_iouh'].item()
        return losses, stats

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)
        self.objective.train(mode)

    def eval(self):
        """ Set network to eval mode"""
        self.train(False)

    '''added by chenxin to fix bn'''
    def fix_bns(self):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        net.featurefusion_network.apply(self.fix_bn)
        net.class_embed.apply(self.fix_bn)
        net.bbox_embed.apply(self.fix_bn)
        net.input_proj.apply(self.fix_bn)
        net.backbone.apply(self.fix_bn)
        if hasattr(net, 'mask_head'):
            net.iou_embed.apply(self.fix_bn)
    '''added by chenxin to fix bn'''

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            # print(classname)
            m.eval()

