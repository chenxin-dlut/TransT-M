import sys
import os
env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from pytracking.VOT2021.transt_seg_class import run_vot_exp


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# run_vot_exp('dimp','dimp50_vot19','SEbcm',0.60,VIS=False)
net_path = '/home/cx/cx1/work_dir/work_dir_seg_v3_8GPU_N2_fixbn/checkpoints/ltr/transt/transt_ddp/TransTsegm_ep0110.pth.tar'
save_root = '/home/cx/cx1/TransT_experiments/vot2021/TransTN4seg_v3_8GPU_thres50_80e_consp'
run_vot_exp('transt',window=0.55,threshold=0.5,net_path=net_path,save_root=save_root,VIS=False)
# run_vot_exp('dimp','super_dimp','ARcm_coco_seg_only_mask_384',0.65,VIS=True)