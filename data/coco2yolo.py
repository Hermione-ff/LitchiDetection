import sys
sys.path.append(r'D:\project\LitchiDetection\ultralytics-main\ultralytics')

from ultralytics.data.converter import convert_coco

convert_coco(labels_dir=r'D:\project\LitchiDetection\data\trainsult', use_segments=False)