from __future__ import absolute_import

import numpy as np
import math
import torchvision.transforms.functional as tvisf
import cv2
import torch
import torch.nn.functional as F
import time


class Tracker(object):

    def __init__(self, name, net, mask, window_penalty=0.49, penalty_k=0,
                 iou_alpha = 0, update_threshold=0.9, mask_threshold=0.5,
                 exemplar_size=128, instance_size=256):
        self.name = name
        self.net = net
        self.num_template = 2
        self.window_penalty = window_penalty
        self.penalty_k = penalty_k
        self.iou_alpha = iou_alpha
        self.mask_threshold = mask_threshold
        self.update_threshold = update_threshold
        self.exemplar_size = exemplar_size
        self.instance_size = instance_size
        self.mask = mask

    def _convert_score(self, score):

        score = score.permute(2, 1, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 0].cpu().numpy()
        return score

    def _convert_bbox(self, delta):

        delta = delta.permute(2, 1, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        return delta

    def _convert_iou(self, delta):

        delta = delta.squeeze(0).squeeze(1)
        delta = delta.data.cpu().numpy()

        return delta

    def _convert_mask(self, delta):

        delta = delta.squeeze(0).squeeze(0)
        delta = delta.data.cpu().numpy()

        return delta

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: rgb based image
            pos: center position
            model_sz: exemplar size
            original_sz: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        im_patch = im_patch.cuda()
        return im_patch

    def map_mask_back(self, im, center_pos, instance_size, s_x, mask, mode=cv2.BORDER_REPLICATE):
        """ Extracts a crop centered at target_bb box, of size search_area_factor times target_bb(Both height and width)

        args:
            im - cv image
            target_bb - target box [x, y, w, h]
            search_area_factor - Ratio of crop size to target size
            output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

        returns:
            cv image - extracted crop
            float - the factor by which the crop has been resized to make the crop size equal output_size
        """
        H, W = (im.shape[0], im.shape[1])
        base = np.zeros((H, W))
        x_center, y_center = center_pos.tolist()

        # Crop image

        if s_x < 1 or s_x < 1:
            raise Exception('Too small bounding box.')
        c = (s_x + 1) / 2

        x1 = int(np.floor(x_center - c + 0.5))
        x2 = int(x1 + s_x - 1)

        y1 = int(np.floor(y_center - c + 0.5))
        y2 = int(y1 + s_x -1)

        x1_pad = int(max(0., -x1))
        y1_pad = int(max(0., -y1))
        x2_pad = int(max(0., x2 - W + 1))
        y2_pad = int(max(0., y2 - H + 1))

        '''pad base'''
        base_padded = cv2.copyMakeBorder(base, y1_pad, y2_pad, x1_pad, x2_pad, mode)
        '''Resize mask'''
        mask_rsz = cv2.resize(mask, (s_x, s_x))
        '''fill region with mask'''
        base_padded[y1 + y1_pad:y2 + y1_pad + 1, x1 + x1_pad:x2 + x1_pad + 1] = mask_rsz.copy()
        '''crop base_padded to get final mask'''
        final_mask = base_padded[y1_pad:y1_pad + H, x1_pad:x1_pad + W]
        assert (final_mask.shape == (H, W))
        return final_mask

    def constraint_mask(self, mask, bbox):
        """
        mask: shape (H, W)
        bbox: list [x1, y1, w, h]
        """
        x1 = np.int(np.floor(bbox[0]))
        y1 = np.int(np.floor(bbox[1]))
        x2 = np.int(np.ceil(bbox[0]+bbox[2]))
        y2 = np.int(np.ceil(bbox[1]+bbox[3]))
        mask[0:y1+1,:] = 0
        mask[y2:,:] = 0
        mask[:,0:x1+1] = 0
        mask[:,x2:] = 0
        return mask

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.net.initialize()
        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        tic = time.time()
        hanning = np.hanning(32)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        # Initialize
        self.initialize_features()
        bbox = info['init_bbox']
        self.center_pos = np.array([bbox[0] + bbox[2] / 2,
                                    bbox[1] + bbox[3] / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # while(True):
        #     cv2.imshow('image', image)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break


        # calculate z crop size
        w_z = self.size[0] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_z = self.size[1] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_z = math.ceil(math.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(image, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(image, self.center_pos,
                                    self.exemplar_size,
                                    s_z, self.channel_average)

        # normalize
        z_crop = z_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.inplace = False
        z_crop[0] = tvisf.normalize(z_crop[0], self.mean, self.std, self.inplace)

        # initialize template feature
        template = self.net.template(z_crop)
        self.templates_list = [template] * self.num_template
        out = {'time': time.time() - tic}
        return out

    def track(self, image, info: dict = None) -> dict:
        # calculate x crop size
        w_x = self.size[0] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_x = self.size[1] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_x = math.ceil(math.sqrt(w_x * h_x))

        # get crop
        x_crop_ori = self.get_subwindow(image, self.center_pos,
                                        self.instance_size,
                                        round(s_x), self.channel_average)

        # normalize
        x_crop = x_crop_ori.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        x_crop[0] = tvisf.normalize(x_crop[0], self.mean, self.std, self.inplace)

        # track
        mask_flag = self.mask
        # outputs = self.net.track_seg(x_crop, mask=mask_flag)
        # outputs = self.net.track_seg(x_crop, self.templates_list, mask=mask_flag)
        outputs = self.net.track(x_crop, self.templates_list)
        score = self._convert_score(outputs['pred_logits'])
        pred_bbox = self._convert_bbox(outputs['pred_boxes'])
        iou = self._convert_iou(outputs['pred_iouh'])
        # def change(r):
        #     return np.maximum(r, 1. / r)
        #
        # def sz(w, h):
        #     pad = (w + h) * 0.5
        #     return np.sqrt((w + pad) * (h + pad))
        #
        # # scale penalty
        # s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
        #              (sz(self.size[0]/s_x, self.size[1]/s_x)))
        #
        # # aspect ratio penalty
        # r_c = change((self.size[0]/self.size[1]) /
        #              (pred_bbox[2, :]/pred_bbox[3, :]))
        # penalty = np.exp(-(r_c * s_c - 1) * self.penalty_k)
        # pscore = penalty * score

        pscore = score
        # pscore = (pscore ** (1-self.iou_alpha)) * (iou ** (self.iou_alpha))
        # window penalty
        pscore = pscore * (1 - self.window_penalty) + \
                 self.window * self.window_penalty

        best_idx = np.argmax(pscore)
        # print(iou[best_idx])
        bbox = pred_bbox[:, best_idx]
        bbox = bbox * s_x
        cx = bbox[0] + self.center_pos[0] - s_x / 2
        cy = bbox[1] + self.center_pos[1] - s_x / 2
        width = bbox[2]
        height = bbox[3]

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, image.shape[:2])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        if mask_flag == True:
            pred_mask = self._convert_mask(outputs['pred_masks'])
            pred_mask = self.map_mask_back(image, self.center_pos, self.instance_size, s_x, pred_mask,
                                      mode=cv2.BORDER_CONSTANT)
            final_mask = (pred_mask > self.mask_threshold).astype(np.uint8)
            final_mask = self.constraint_mask(final_mask, bbox)

        # update state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        out = {'target_bbox': bbox,
               'best_score': pscore[best_idx]}
        if mask_flag == True:
            out['target_mask'] = final_mask
        # print(iou[best_idx])
        if iou[best_idx] > self.update_threshold:
            self.update(image, out)
        return out

    def update(self, image, info: dict):
        bbox = info['target_bbox']
        self.center_pos = np.array([bbox[0] + bbox[2] / 2,
                                    bbox[1] + bbox[3] / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # while(True):
        #     cv2.imshow('image', image)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        # calculate z crop size
        w_z = self.size[0] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_z = self.size[1] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_z = math.ceil(math.sqrt(w_z * h_z))

        # calculate channle average

        # get crop
        z_crop = self.get_subwindow(image, self.center_pos,
                                    self.exemplar_size,
                                    s_z, self.channel_average)

        # normalize
        z_crop = z_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.inplace = False
        z_crop[0] = tvisf.normalize(z_crop[0], self.mean, self.std, self.inplace)

        # initialize template feature
        template = self.net.template(z_crop)
        self.templates_list.pop(1)
        self.templates_list.append(template)
