import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train/your name/weights/best.pt')
    model.val(data='ultralytics/cfg/datasets/VisDrone.yaml',
              split='test',
              imgsz=640,
              batch=1,
            #   save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='your name',
              )
