import sys
import os
env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from pytracking.VOT2021.transt_seg_class import run_vot_exp


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# run_vot_exp('dimp','dimp50_vot19','SEbcm',0.60,VIS=False)
net_path = '/home/cx/cx1/work_dir/work_dir_mt_2t_e464_iouh_seg/checkpoints/ltr/transt/transt_iouh_seg_ddp/TransTiouhsegm_ep0090.pth.tar'
save_root = '/home/cx/cx1/TransT_experiments/vot2021/visualize/mt_2t'
run_vot_exp('transt',mask=True, window=0.50,penalty_k=0,update_threshold=10, mask_threshold=0.5,net_path=net_path,save_root=save_root,VIS=True)
# run_vot_exp('dimp','super_dimp','ARcm_coco_seg_only_mask_384',0.65,VIS=True)