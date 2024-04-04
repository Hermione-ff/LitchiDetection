from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image
# 初始化检测模型
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=r'D:\project\LitchiDetection\yolov8_recode\train2\weights\best.pt',
    confidence_threshold=0.3,
    device="cuda:0",  # or 'cuda:0'
)
image = r"E:\毕业论文\leaves\train\DJI_20231217140700_0023_Z_Area02-60degrees-sampling20x-J47-H7-W.jpeg"


result = get_sliced_prediction(
    image,
    detection_model,
    slice_height = 1600,
    slice_width = 1600,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2,
    perform_standard_pred = True,
)

res = result.export_visuals()
print(result.count_predictions_by_class())
im = Image.fromarray(res['image'])
im.show()


# # result是SAHI的PredictionResult对象，可获得推理时间，检测图像，检测图像尺寸，检测结果
# # 查看标注框，可以用于保存为其他格式
# for pred in result.object_prediction_list:
#     bbox = pred.bbox  # 标注框BoundingBox对象，可以获得边界框的坐标、面积
#     category = pred.category  # 类别Category对象，可获得类别id和类别名
#     score = pred.score.value  # 预测置信度
#
# # 保存文件结果
# export_dir = ""
# file_name = "res1"
# result.export_visuals(export_dir=export_dir, file_name=file_name)
# # 结果导出为coco标注形式
# coco_anno = result.to_coco_annotations()
# # 结果导出为coco预测形式
# coco_pred = result.to_coco_predictions()

# # 展示结果
# from PIL import Image
# import os
# image_path = os.path.join(export_dir,file_name+'.png')
# img = Image.open(image_path).convert('RGB')
# img

