from ultralytics import YOLO
# 假设你的YOLO模型是YOLOv4或YOLOv5，它们有相应的模型定义文件
# 例如，对于YOLOv5，你可能会有一个yolov5s.yaml文件，你可以从这个文件加载模型结构
# 如果你有一个预训练的pt文件，你也可以直接从pt文件加载模型结构和权重
from ultralytics.nn.tasks import DetectionModel

# 加载模型权重
model = YOLO(r'D:\project\LitchiDetection\yolov8_recode\train1\weights\best.pt')  # 替换为你的YOLO模型pt文件路径


# 打印模型结构

# 模型网络结构配置文件路径
yaml_path = 'ultralytics/cfg/models/v8/yolov8s.yaml'
# 改进的模型结构路径
# yaml_path = 'ultralytics/cfg/models/v8/yolov8n-CBAM.yaml'
# 传入模型网络结构配置文件cfg, nc为模型检测类别数
DetectionModel(cfg=yaml_path,nc=2)

