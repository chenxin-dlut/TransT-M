import sys
import os
env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from pytracking.VOT2021.transt_seg_class import run_vot_exp

net_path = '/Path_to/transtm_vot.pth' # path to transtm_vot model
save_root = ''
run_vot_exp('transt',mask=True, window=0.574,penalty_k=0.246,update_threshold=0.869, mask_threshold=0.60,
            net_path=net_path,save_root=save_root,VIS=False)