#YOLOV8使用

##安装与使用
- 源码安装
```TXT
https://github.com/ultralytics/ultralytics
https://github.com/obss/sahi/tree/main
pip install -e .
使用前加 
import sys
sys.path.append(r'D:\project\LitchiDetection\ultralytics-main\ultralytics')
sys.path.append(r'D:\project\LitchiDetection\sahi-main\'sahi')
```
- 预测路径修改
```text
/ultralytics/cfg/default.yaml
添加字段  save_dir: ./runs/train1 # 自己设置路径
```


##数据集-sahi

###初始数据集
- 标注完数据后,将图片与json文件全部放到一个文件夹里。
###划分数据集
- 使用脚本将文件夹里的数据按7:2:1划分为训练集,验证集,测试集,脚本文件是data/split_datasets.py
- 划分数据集后对三个文件夹先进行图片与json分开,在对json进行转coco格式，使用labelme2coco.py
###切分数据集
- 对coco格式的标注文件进行sahi切片，选择切片大小为1600X1600,使用slice_coco.py
- 最后,转回yolo格式,使用coco2yolo.py
###完成数据集
- 最后将切片后的数据集放入dataset/Litchi中。
- 数据集配置文件在data/dataset中


##训练
- 每一次的训练脚本,参数,结果都放在yolov8/recode下
- 模型的配置参数放在cfg/v8/yolov8-Litchi.yaml中


##模型修改
###增添PolarizedSelfAttention注意力机制
- ultralytics/nn/modules/conv.py下增加注意力机制类代码,并在init中添加
- ultralytics/nn/task.py添加任务,函数def parse_model(d, ch, verbose=True): # model_dict, input_channels(3)进行修改
- 修改cfg的配置文件

###替换Lion优化器
- 增添优化器到ultralytics/engine
- 在trainer.py中导入优化器

###替换MPDioU损失函数
- 修改ultralytics/yolo/utils/metrics.py中的bbox_iou函数
- 修改在ultralytics/yolo/utils/loss.py中BboxLoss类的forward函数：
- 由于YOLOv8中在标签分配规则中也有用到bbox_iou的函数，所以需要修改ultralytics/yolo/utils/tal.py的TaskAlignedAssigner类中的get_box_metrics函数

##替换Quality focal loss函数
- 在yolo/utils/loss.py下添加新的focal_loss的代码
- 在loss.py中找到DetectionLoss类定义focal_loss变量



#streamlit可视化
- 修改sahi-main/sahi/predicition.py下的PredictionResult类的export_visuals方法以及增加数类别操作,这样修改能返回一个包含处理后图像和时间的字典,以及一个统计类别的字典
```txt
def count_predictions_by_class(self):
    class_counts = {}
    for object_prediction in self.object_prediction_list:
        category_id = object_prediction.category.id
        if category_id in class_counts:
            class_counts[category_id] += 1
        else:
            class_counts[category_id] = 1
    return class_counts

def export_visuals(
    self,
    export_dir: str = None,
    text_size: float = None,
    rect_th: int = None,
    hide_labels: bool = False,
    hide_conf: bool = False,
    file_name: str = "prediction_visual",
):
    """

    Args:
        export_dir: directory for resulting visualization to be exported
        text_size: size of the category name over box
        rect_th: rectangle thickness
        hide_labels: hide labels
        hide_conf: hide confidence
        file_name: saving name
    Returns:

    """
    if export_dir:
        Path(export_dir).mkdir(parents=True, exist_ok=True)
    res = visualize_object_predictions(
        image=np.ascontiguousarray(self.image),
        object_prediction_list=self.object_prediction_list,
        rect_th=rect_th,
        text_size=text_size,
        text_th=None,
        color=None,
        hide_labels=hide_labels,
        hide_conf=hide_conf,
        output_dir=export_dir,
        file_name=file_name,
        export_format="png",
    )
    return res
```
- 主页面布局都在app.py里
- 模型地址,参数等在config.py里,模型参数放入weights里