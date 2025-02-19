import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov5.yaml')
    # model.load('yolo11n.pt') # 是否加载预训练权重

    model.train(data='Flappybird.yaml', #替换你自己数据集的Yaml文件地址
                cache=False,
                imgsz=640,
                epochs=20,        #训练轮数
                single_cls=False,  # 是否是单类别检测
                batch=16,
                close_mosaic=0,
                workers=0,
                device='cpu',      #选择'cpu'或者'0'
                optimizer='SGD', # using SGD
                # resume='runs/train/exp21/weights/last.pt', # 续训设置last.pt的地址
                amp=True,
                project='runs/train',
                name='exp',
                )
    
    metrics = model.val()  # 运行验证并返回评估指标