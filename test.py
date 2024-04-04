import sys
sys.path.append(r'D:\project\LitchiDetection\ultralytics-main\ultralytics')

#from ultralytics import YOLO


# yolo = YOLO("./ultralytics-main/yolov8n.pt", task="detect")
# result = yolo(source="./ultralytics-main/ultralytics/assets/bus.jpg", save=True)


# import torch
#
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())


# import sahi
# # 打印sahi版本
# print(sahi.__version__)



from ultralytics import RTDETR, YOLO
from ultralytics.utils import (
    ASSETS,
    DEFAULT_CFG,
    DEFAULT_CFG_PATH,
    LINUX,
    MACOS,
    ONLINE,
    ROOT,
    WEIGHTS_DIR,
    WINDOWS,
    Retry,
    checks,
    is_dir_writeable,
)


CFG = r"D:\project\LitchiDetection\ultralytics-main\ultralytics\cfg\models\v8\yolov8s-Litchi.yaml"
SOURCE = ASSETS / "bus.jpg"



def test_model_forward():
    """Test the forward pass of the YOLO model."""
    model = YOLO(CFG)
    model(source=None, imgsz=32, augment=True)  # also test no source and augment