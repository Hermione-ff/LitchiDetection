# -*- coding: utf-8 -*-
from pathlib import Path
import sys

# 获取脚本当前的绝对路径
file_path = Path(__file__).resolve()

# 返回当前脚本的父目录路径
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# 获取根目录相对于当前工作目录的相对路径。
ROOT = root_path.relative_to(Path.cwd())


# Source
SOURCES_LIST = ["Image"]


# DL model config
DETECTION_MODEL_DIR = ROOT / 'weights'
YOLOv8n = DETECTION_MODEL_DIR / "yolov8n.pt"
YOLOv8s = DETECTION_MODEL_DIR / "yolov8s.pt"
YOLOv8s_Litchi_1600 = DETECTION_MODEL_DIR / "yolov8s-Litchi-1600.pt"
YOLOv8s_Litchi_1334 = DETECTION_MODEL_DIR / "yolov8s-Litchi-1334.pt"

DETECTION_MODEL_LIST = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8s-Litchi-1600.pt",
    "yolov8s-Litchi-1334.pt",
    ]


OBJECT_COUNTER = None
OBJECT_COUNTER1 = None