import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train-tinyperson/your name/weights/best.pt')
    model.val(data='ultralytics/cfg/datasets/tinyperson.yaml',
              split='test',
              imgsz=640,
              batch=1,
            #   save_json=True, # if you need to cal coco metrice
              project='runs/val-tinyperson',
              name='your name',
              )
