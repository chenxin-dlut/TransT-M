# TransT-M - High-performance Transformer Tracking
Official implementation of the TransT-M, including training code and trained models.
[Models](https://drive.google.com/drive/folders/1-DtTGZE1Q7MAWX9ufv_xif1SCA_zQE0H?usp=sharing)

## Installation
This document contains detailed instructions for installing the necessary dependencied for **TransT-M**. The instructions 
have been tested on Ubuntu 18.04 system.

#### Install dependencies
* Create and activate a conda environment 
```bash
conda create -n transt python=3.7
conda activate transt
```  
* Install PyTorch
```bash
conda install -c pytorch pytorch=1.5 torchvision
```  

* Install other packages
```bash
conda install matplotlib pandas tqdm
pip install opencv-python tb-nightly visdom scikit-image tikzplotlib gdown
conda install cython scipy
pip install pycocotools jpeg4py
pip install wget
pip install shapely==1.6.4.post2
```  
* Setup the environment                                                                                                 
Create the default environment setting files.

```bash
# Change directory to <PATH_of_TransT>
cd TransT-M

# Environment settings for pytracking. Saved at pytracking/evaluation/local.py
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

# Environment settings for ltr. Saved at ltr/admin/local.py
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
```
You can modify these files to set the paths to datasets, results paths etc.
* Add the project path to environment variables  
Open ~/.bashrc, and add the following line to the end. Note to change <path_of_TransT> to your real path.
```
export PYTHONPATH=<path_of_TransT>:$PYTHONPATH
```
* Download the pre-trained networks 
Download the network for [TransT-M](https://drive.google.com/drive/folders/1-DtTGZE1Q7MAWX9ufv_xif1SCA_zQE0H?usp=sharing)
and put it in the directory set by "network_path" in "pytracking/evaluation/local.py". By default, it is set to 
pytracking/networks.

## Quick Start
#### TRAINING
* Modify [local.py](ltr/admin/local.py) to set the paths to datasets, results paths etc.
* Runing the following commands to train the TransT-M. You can customize some parameters by modifying the settings in [transt](ltr/train_settings/transt/)
1. Train the base model of TransT-M
```bash
conda activate transt
cd TransT-M/ltr
python -m torch.distributed.launch --nproc_per_node 8 run_training_multigpu.py transt transt
```  
2. Train the iou head of TransT-M, you should set a new workspace_dir in [local.py](ltr/admin/local.py) and modify the settings.transt_path in [transt_iou.py](ltr/train_settings/transt/transt_iou.py) to the path of a trained base transt model
```bash
python -m torch.distributed.launch --nproc_per_node 8 run_training_multigpu.py transt transt_iou
```  

3. Train the segmentation branch of TransT-M, you should set a new workspace_dir in [local.py](ltr/admin/local.py) and modify the settings.transt_path in [transt_iou_seg.py](ltr/train_settings/transt/transt_iou_seg.py) to the path of a trained transt_iou model
```bash
python -m torch.distributed.launch --nproc_per_node 8 run_training_multigpu.py transt transt_iou_seg
```  
#### Evaluation
* We integrated [PySOT](https://github.com/STVIR/pysot) for evaluation
    You need to specify the path of the model and dataset in the following files: [test_got.py](pysot_toolkit/test_got.py), [test_lasot.py](pysot_toolkit/test_lasot.py), [test_nfs.py](pysot_toolkit/test_nfs.py), [test_otb.py](pysot_toolkit/test_otb.py), [test_tracking.py](pysot_toolkit/test_tracking.py), [test_uav.py](pysot_toolkit/test_uav.py)
    ```python
    net_path = '/path_to_model' #Absolute path of the model
    dataset_root= '/path_to_datasets' #Absolute path of the datasets
    ```  
    You need to specify the path of dataset in [eval.py](pysot_toolkit/eval.py)
    ```python
    root = '/path_to_datasets' #Absolute path of the datasets
    ```  

    Then run the following commands
    ```bash
    conda activate TransT
    cd TransT-M
    python -u pysot_toolkit/test_lasot.py --dataset LaSOT #test tracker
    python pysot_toolkit/eval.py --tracker_path pysot_toolkit/results/ --dataset LaSOT --num 1 #eval tracker

    python -u pysot_toolkit/test_got.py --dataset GOT-10k #test tracker

    python -u pysot_toolkit/test_trackingnet.py --dataset Tracking #test tracker
  
    python -u pysot_toolkit/test_nfs.py --dataset NFS #test tracker
    python pysot_toolkit/eval.py --tracker_path pysot_toolkit/results/ --dataset NFS --num 1 #eval tracker

    python -u pysot_toolkit/test_uav.py --dataset UAV #test tracker
    python pysot_toolkit/eval.py --tracker_path pysot_toolkit/results/ --dataset UAV --num 1 #eval tracker

    python -u pysot_toolkit/test_otb.py --dataset OTB #test tracker
    python pysot_toolkit/eval.py --tracker_path pysot_toolkit/results/ --dataset OTB --num 1 #eval tracker
    ```  
* For evaluation on VOT2021, run the following commands. You should modify the paths in [trackers.ini](vot2021_workspace%2Ftrackers.ini), and the net path in [transt_VOT2021.py](pytracking%2FVOT2021%2Ftranst_VOT2021.py)
    ```bash
    cd TransT-M/vot2021_workspace
    vot evaluate TransT_M
    ``` 

## Acknowledgement
This is a modified version of the python framework [PyTracking](https://github.com/visionml/pytracking) based on **Pytorch**, 
also borrowing from [PySOT](https://github.com/STVIR/pysot) and [GOT-10k Python Toolkit](https://github.com/got-10k/toolkit). 
We would like to thank their authors for providing great frameworks and toolkits.
## Contact
* Xin Chen (email:chenxin3131@mail.dlut.edu.cn)
