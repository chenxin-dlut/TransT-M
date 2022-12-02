import torch
import torchvision.transforms as transforms
from pytracking import TensorDict
import ltr.data.processing_utils as prutils


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), search_transform=None, template_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if search_transform or
                                template_transform is None.
            search_transform - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            template_transform  - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the search and template images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'search': transform if search_transform is None else search_transform,
                          'static_template':  transform if template_transform is None else template_transform,
                          'dynamic_template':  transform if template_transform is None else template_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class TransTProcessing(BaseProcessing):
    """ The processing class used for training TransT. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument search_sz.

    """

    def __init__(self, search_area_factor, template_area_factor, search_sz, temp_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region relative to the target size.
            template_area_factor - The size of the template region relative to the template target size.
            search_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            temp_sz - An integer, denoting the size to which the template region is resized. The search region is always
                      square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.template_area_factor = template_area_factor
        self.search_sz = search_sz
        self.temp_sz = temp_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'search' or 'template' indicating search or template data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.sum() * 0.5 * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)
        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'search_images', template_images', 'search_anno', 'template_anno'
        returns:
            TensorDict - output data block with following fields:
                'search_images', 'template_images', 'search_anno', 'template_anno'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'] = self.transform['joint'](image=data['template_images'], bbox=data['template_anno'])
            data['search_images'], data['search_anno'] = self.transform['joint'](image=data['search_images'], bbox=data['search_anno'], new_roll=False)

        for s in ['search', 'template']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num search/template frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            if s == 'search':
                crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                           self.search_area_factor, self.search_sz)
            elif s == 'template':
                crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                           self.template_area_factor, self.temp_sz)
            else:
                raise NotImplementedError

            # Apply transforms
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)
        data['template_images'] = data['template_images'].squeeze()
        data['search_images'] = data['search_images'].squeeze()
        data['template_anno'] = data['template_anno'].squeeze()
        data['search_anno'] = data['search_anno'].squeeze()
        return data


class TransTMaskProcessing(TransTProcessing):
    """ The processing class used for training TransT. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument search_sz.

    """

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'search_images', template_images', 'search_anno', 'template_anno'
        returns:
            TensorDict - output data block with following fields:
                'search_images', 'template_images', 'search_anno', 'template_anno'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['static_template_images'], data['static_template_anno'] = self.transform['joint'](image=data['static_template_images'],
                                                                                                   bbox=data['static_template_anno'])
            if data['dynamic_template'] == True:
                data['dynamic_template_images'], data['dynamic_template_anno'] = self.transform['joint'](image=data['dynamic_template_images'],
                                                                                                         bbox=data['dynamic_template_anno'],
                                                                                                         new_roll=False)
            data['search_images'], data['search_anno'] = self.transform['joint'](image=data['search_images'],
                                                                                 bbox=data['search_anno'],
                                                                                 new_roll=False)

        s_index = ['search', 'static_template']
        if data['dynamic_template'] == True:
            s_index.append('dynamic_template')
        for s in s_index:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num search/template frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            if s == 'search':
                crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                               self.search_area_factor, self.search_sz)
            elif s == 'static_template':
                # crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                #                                                self.template_area_factor, self.temp_sz)
                crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                               self.template_area_factor, self.temp_sz)
            elif s == 'dynamic_template':
                # crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                #                                                self.template_area_factor, self.temp_sz)
                crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                               self.template_area_factor, self.temp_sz)
            else:
                raise NotImplementedError

            # Apply transforms
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)
            #mask_crops:(256, 256),crops:(256, 256, 3)
            if s == 'search':
                mask_crops, _ = prutils.jittered_center_crop(data[s + '_masks'], jittered_anno, data[s + '_anno'],
                                                               self.search_area_factor, self.search_sz)
            elif s == 'static_template':
                mask_crops, _ = prutils.jittered_center_crop(data[s + '_masks'], jittered_anno, data[s + '_anno'],
                                                               self.template_area_factor, self.temp_sz)
            elif s == 'dynamic_template':
                mask_crops, _ = prutils.jittered_center_crop(data[s + '_masks'], jittered_anno, data[s + '_anno'],
                                                               self.template_area_factor, self.temp_sz)
            else:
                raise NotImplementedError
            data[s + '_masks'] = [self.mask_np2torch(x) for x in mask_crops]

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        data['search_images'] = data['search_images'].squeeze(0) # torch.Size([3, Hs, Ws])
        data['search_masks'] = data['search_masks'].squeeze(0) #torch.Size([1, Hs, Ws])
        data['search_anno'] = data['search_anno'].squeeze(0) #torch.Size([4])

        data['template_images'] = data['static_template_images'] # torch.Size([1, 3, Ht, Wt])
        data['template_masks'] = data['static_template_masks'] #torch.Size([1, 1, Ht, Wt]) #template的mask其实可以去掉，没用到
        data['template_anno'] = data['static_template_anno'] #torch.Size([1, 4])

        if data['dynamic_template'] == True:
            data['template_images'] = torch.cat((data['template_images'], data['dynamic_template_images']), 0)#torch.Size([number, 3, Ht, Wt])
            data['template_masks'] = torch.cat((data['template_masks'], data['dynamic_template_masks']), 0) #torch.Size([number, 1, 128, 128])
            data['template_anno'] = torch.cat((data['template_anno'], data['dynamic_template_anno']), 0) #torch.Size([number, 4])

        return data

    def mask_np2torch(self, mask_np):
        return torch.from_numpy(mask_np.transpose((2, 0, 1))).float()