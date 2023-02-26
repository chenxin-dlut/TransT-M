class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/cx/cx1/work_dir/transt'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = '/home/cx/cx3/LaSOTBenchmark'
        self.got10k_dir = '/home/cx/cx3/GOT10K/train'
        self.trackingnet_dir = '/home/cx/cx1/TrackingNet'
        self.coco_dir = '/home/cx/cx3/COCO'
        self.lvis_dir = ''
        self.sbd_dir = ''



        self.imagenet_dir = '/home/cx/cx3/ILSVRC2015'
        self.imagenetdet_dir = '/home/cx/cx3/Imagenet_DET/ILSVRC'
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.youtube_vos_dir = '/home/cx/cx3/Youtube-VOS'
        self.saliency_dir = '/home/cx/cx3/saliency/MERGED'

