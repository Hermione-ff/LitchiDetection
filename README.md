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
- 划分数据集后对三个文件夹先进行图片与json分开(创建两个文件夹,label和image,都按之前分好的数据集),在对json进行转coco格式，使用labelme2coco.py,最终形成三个json文件
###切分数据集
- 对coco格式的标注文件进行sahi切片，选择切片大小为1334X1334,使用slice_coco.py
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
- 经过测试发现效果不如之前,换回交叉熵损失

##在C2f中加入Biformer
- pip install einops
- 在nn/modules中加入biformer.py
- 在block中添加C2fattention2以及在init中导入
- 在task的parse_model中注册
- 修改cfg的配置文件
- 效果不如之前,用回原来的c2f模块

##添加SegNext_attention
- 在conv.py中添加segnext_attention
- 在init中注册
- 在task中引入,parse_model中注册



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
- 修改sahi-main/sahi/predict.py下的get_sliced_prediction函数,使其可以通过输入参数来控制是否返回切片结果
```txt
def get_sliced_prediction(
    image,
    detection_model=None,
    output_file_name=None,  # ADDED OUTPUT FILE NAME TO (OPTIONALLY) SAVE SLICES
    interim_dir="slices/",  # ADDED INTERIM DIRECTORY TO (OPTIONALLY) SAVE SLICES
    slice_height: int = None,
    slice_width: int = None,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    perform_standard_pred: bool = True,
    postprocess_type: str = "GREEDYNMM",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    postprocess_class_agnostic: bool = False,
    verbose: int = 1,
    merge_buffer_length: int = None,
    auto_slice_resolution: bool = True,
    get_slice_result = False,
) -> PredictionResult:
    """
    Function for slice image + get predicion for each slice + combine predictions in full image.

    Args:
        image: str or np.ndarray
            Location of image or numpy image matrix to slice
        detection_model: model.DetectionModel
        slice_height: int
            Height of each slice.  Defaults to ``None``.
        slice_width: int
            Width of each slice.  Defaults to ``None``.
        overlap_height_ratio: float
            Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        overlap_width_ratio: float
            Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        perform_standard_pred: bool
            Perform a standard prediction on top of sliced predictions to increase large object
            detection accuracy. Default: True.
        postprocess_type: str
            Type of the postprocess to be used after sliced inference while merging/eliminating predictions.
            Options are 'NMM', 'GRREDYNMM' or 'NMS'. Default is 'GRREDYNMM'.
        postprocess_match_metric: str
            Metric to be used during object prediction matching after sliced prediction.
            'IOU' for intersection over union, 'IOS' for intersection over smaller area.
        postprocess_match_threshold: float
            Sliced predictions having higher iou than postprocess_match_threshold will be
            postprocessed after sliced prediction.
        postprocess_class_agnostic: bool
            If True, postprocess will ignore category ids.
        verbose: int
            0: no print
            1: print number of slices (default)
            2: print number of slices and slice/prediction durations
        merge_buffer_length: int
            The length of buffer for slices to be used during sliced prediction, which is suitable for low memory.
            It may affect the AP if it is specified. The higher the amount, the closer results to the non-buffered.
            scenario. See [the discussion](https://github.com/obss/sahi/pull/445).
        auto_slice_resolution: bool
            if slice parameters (slice_height, slice_width) are not given,
            it enables automatically calculate these params from image resolution and orientation.

    Returns:
        A Dict with fields:
            object_prediction_list: a list of sahi.prediction.ObjectPrediction
            durations_in_seconds: a dict containing elapsed times for profiling

    参数：

    image: 输入的图像，可以是图像的路径或者是一个numpy数组。
    detection_model: 检测模型，用于对图像进行物体检测。
    output_file_name: （可选）切片后的图像保存路径。
    interim_dir: （可选）中间切片图像保存的目录。
    slice_height 和 slice_width: 每个切片的高度和宽度。
    overlap_height_ratio 和 overlap_width_ratio: 切片之间的重叠部分的比例。
    perform_standard_pred: 是否在切片预测之后执行标准预测，以增加对大物体的检测准确性。
    postprocess_type: 切片预测后进行合并/消除预测时要使用的后处理类型。
    postprocess_match_metric: 预测匹配时使用的度量标准。
    postprocess_match_threshold: 预测匹配阈值。
    postprocess_class_agnostic: 是否忽略类别标识。
    verbose: 是否输出详细信息。
    merge_buffer_length: 用于切片预测的缓冲区长度，适用于内存较低的情况。
    auto_slice_resolution: 是否自动计算切片参数（高度和宽度）。
    函数内部逻辑：

    首先，函数会调用 slice_image 函数对输入图像进行切片，并获取切片结果。
    然后，根据切片结果逐个对切片进行预测，将预测结果存储在 object_prediction_list 中。
    如果指定了 merge_buffer_length，并且 object_prediction_list 中的预测结果数量超过了缓冲区长度，则会调用后处理函数对预测结果进行合并。
    如果 perform_standard_pred 为真且切片数量大于1，则会对完整图像进行一次标准预测，并将其结果添加到 object_prediction_list 中。
    最后，对 object_prediction_list 中的所有预测结果进行最终的合并操作，并返回包含图像、预测结果和执行时间的 PredictionResult 对象。
    这个函数的核心思想是通过切片的方式来处理大尺寸图像，以降低计算复杂度，并在切片预测的基础上，通过后处理来合并和消除重叠的预测结果，从而得到最终的物体检测结果。
    """

    # for profiling
    durations_in_seconds = dict()

    # currently only 1 batch supported
    num_batch = 1

    # create slices from full image
    time_start = time.time()
    slice_image_result = slice_image(
        image=image,
        output_file_name=output_file_name,  # ADDED OUTPUT FILE NAME TO (OPTIONALLY) SAVE SLICES
        output_dir=interim_dir,  # ADDED INTERIM DIRECTORY TO (OPTIONALLY) SAVE SLICES
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        auto_slice_resolution=auto_slice_resolution,
    )
    num_slices = len(slice_image_result)
    time_end = time.time() - time_start
    durations_in_seconds["slice"] = time_end

    # init match postprocess instance
    if postprocess_type not in POSTPROCESS_NAME_TO_CLASS.keys():
        raise ValueError(
            f"postprocess_type should be one of {list(POSTPROCESS_NAME_TO_CLASS.keys())} but given as {postprocess_type}"
        )
    elif postprocess_type == "UNIONMERGE":
        # deprecated in v0.9.3
        raise ValueError("'UNIONMERGE' postprocess_type is deprecated, use 'GREEDYNMM' instead.")
    postprocess_constructor = POSTPROCESS_NAME_TO_CLASS[postprocess_type]
    postprocess = postprocess_constructor(
        match_threshold=postprocess_match_threshold,
        match_metric=postprocess_match_metric,
        class_agnostic=postprocess_class_agnostic,
    )

    # create prediction input
    num_group = int(num_slices / num_batch)
    if verbose == 1 or verbose == 2:
        tqdm.write(f"Performing prediction on {num_slices} number of slices.")
    object_prediction_list = []
    slice_result = []
    # perform sliced prediction
    for group_ind in range(num_group):
        # prepare batch (currently supports only 1 batch)
        image_list = []
        shift_amount_list = []
        for image_ind in range(num_batch):
            image_list.append(slice_image_result.images[group_ind * num_batch + image_ind])
            shift_amount_list.append(slice_image_result.starting_pixels[group_ind * num_batch + image_ind])
        # perform batch prediction
        prediction_result = get_prediction(
            image=image_list[0],
            detection_model=detection_model,
            shift_amount=shift_amount_list[0],
            full_shape=[
                slice_image_result.original_image_height,
                slice_image_result.original_image_width,
            ],
        )
        slice_result.append(prediction_result)
        # convert sliced predictions to full predictions
        for object_prediction in prediction_result.object_prediction_list:
            if object_prediction:  # if not empty
                object_prediction_list.append(object_prediction.get_shifted_object_prediction())

        # merge matching predictions during sliced prediction
        if merge_buffer_length is not None and len(object_prediction_list) > merge_buffer_length:
            object_prediction_list = postprocess(object_prediction_list)

    # perform standard prediction
    if num_slices > 1 and perform_standard_pred:
        prediction_result = get_prediction(
            image=image,
            detection_model=detection_model,
            shift_amount=[0, 0],
            full_shape=None,
            postprocess=None,
        )
        object_prediction_list.extend(prediction_result.object_prediction_list)

    # merge matching predictions
    if len(object_prediction_list) > 1:
        object_prediction_list = postprocess(object_prediction_list)

    time_end = time.time() - time_start
    durations_in_seconds["prediction"] = time_end

    if verbose == 2:
        print(
            "Slicing performed in",
            durations_in_seconds["slice"],
            "seconds.",
        )
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )
    if not get_slice_result:
        return PredictionResult(
            image=image, object_prediction_list=object_prediction_list, durations_in_seconds=durations_in_seconds
        )
    else:
        return PredictionResult(image=image, object_prediction_list=object_prediction_list, durations_in_seconds=durations_in_seconds) , slice_result
```
- 主页面布局都在app.py里
- 模型地址,参数等在config.py里,模型参数放入weights里