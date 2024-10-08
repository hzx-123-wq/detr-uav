import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.rtdetr.distill import RTDETRDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': 'ultralytics/cfg/models/rt-detr/detr-star-soep-dysample-dgcst2.yaml',
        'data':'/home/han/RTDETR-main/ultralytics/cfg/datasets/VisDrone.yaml',
        'imgsz': 640,
        'epochs': 150,
        'batch': 4,
        'workers': 4,
        'cache': True,
        'device': '0',
        'project':'runs/distill',
        'name':'T1-rtdetr-cwd-constant-all',
        
        # distill
        'prune_model': False,
        'teacher_weights':'runs/train/rtdetr-r50-bifpn/weights/best.pt',
        'teacher_cfg': 'ultralytics/cfg/models/rt-detr/rtdetr-r50-bifpn.yaml',
        'kd_loss_type': 'all',
        'kd_loss_decay': 'constant',
        'kd_loss_epoch': 1.0,
        
        'logical_loss_type': 'logical',
        'logical_loss_ratio': 0.25,
        
        'teacher_kd_layers': '3,5,6,7,10,22,25,28',
        'student_kd_layers': '0-1,0-2,0-3,0-4,3,13,16,19',
        'feature_loss_type': 'cwd',
        'feature_loss_ratio': 0.02
    }
    
    model = RTDETRDistiller(overrides=param_dict)
    model.distill()