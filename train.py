import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/detr-star-soep-dysample-dgcst2.yaml')
    #model.load('rtdetr-r50.pt') # loading pretrain weights
    model.train(data='ultralytics/cfg/datasets/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=150,
                batch=4,
                workers=4,
                device='0',
                # resume='', # last.pt path
                project='runs/train',
                name='detr-SOEP-TEST',
                )