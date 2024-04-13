from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image, ImageDraw
from typing import List





def concatenate_images(image_list, image_size,num_rows: int,num_cols: int,spacing: int) -> Image.Image:
    """
    将图像列表拼接成一张大图。

    Args:
        image_list (List[np.ndarray]): 要拼接的图像列表。
        image_size (Tuple[int, int]): 列表中每张图像的尺寸，格式为 (高度, 宽度)。
        num_rows (int): 最终大图的行数。
        num_cols (int): 最终大图的列数。
        spacing (int): 最终大图中图像之间的间隔。

    Returns:
        Image.Image: 拼接后的大图。
    """
    # 计算每个切片的尺寸
    slice_height, slice_width = image_size

    # 计算大图的尺寸
    big_image_width = num_cols * slice_width + (num_cols - 1) * spacing
    big_image_height = num_rows * slice_height + (num_rows - 1) * spacing

    # 创建大图
    big_image = Image.new('RGB', (big_image_width, big_image_height), color='white')

    # 将图像列表中的图像拼接到大图上
    for i, img in enumerate(image_list):
        row = i // num_cols
        col = i % num_cols
        x_offset = col * (slice_width + spacing)
        y_offset = row * (slice_height + spacing)
        big_image.paste(img, (x_offset, y_offset))

    return big_image

if __name__ == '__main__':
    image = r"E:\毕业论文\leaves\train\DJI_20231217140700_0023_Z_Area02-60degrees-sampling20x-J47-H7-W.jpeg"
    # 单张图像切片 slice_image函数源代码位于sahi/slicing.py中
    # SAHI提供slice_image函数以切分单张图片及其标注文件（仅支持coco标注文件），slice_image函数接口介绍如下：
    # 返回SAHI的图像分片结果类SliceImageResult

    # img = Image.open(image).convert('RGB')
    #
    # #切分图像
    # sliced_image_result = slice_image(
    #     image=image,
    #     slice_height=1334,
    #     slice_width=1334,
    #     overlap_height_ratio=0.25,
    #     overlap_width_ratio=0,
    #     verbose=True,
    # )
    # print("原图宽{}，高{}".format(sliced_image_result.original_image_width, sliced_image_result.original_image_height))
    # # 切分后的子图以形式：图像前缀_所在原图顶点坐标来保存文件
    # print("切分子图{}张".format(len(sliced_image_result.filenames)))
    #
    #
    #
    # # 获取切割后的图像列表
    # sliced_images = sliced_image_result.sliced_image_list
    # # 使用匿名函数将sliced_images里的每个对象都取它的image属性
    # image_list = [Image.fromarray(sliced_image.image) for sliced_image in sliced_images]
    # # 计算每个切片的尺寸
    # slice_height, slice_width = sliced_images[0].image.shape[:2]
    #
    # big_image = concatenate_images(image_list=image_list,image_size=(slice_height,slice_width),num_cols=3,num_rows=3,spacing=20)
    # big_image.show()






    # 初始化检测模型,并返回预测大图或者切片结果预测大图
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=r'D:\project\LitchiDetection\yolov8_recode\train6\weights\best.pt',
        confidence_threshold=0.3,
        device="cuda:0",  # or 'cuda:0'
    )
    result , slice_res= get_sliced_prediction(
        image,
        detection_model,
        slice_height = 1334,
        slice_width = 1334,
        overlap_height_ratio = 0.25,
        overlap_width_ratio = 0,
        perform_standard_pred = False, #不用直接对大图进行预测
        get_slice_result=True,
    )

    #预测结果大图
    res  = result.export_visuals()
    print(result.count_predictions_by_class())
    im = Image.fromarray(res['image'])
    im.show()


    #切片结果大图
    # sliced_images = [sli_res.export_visuals() for sli_res in slice_res]
    # image_list = [Image.fromarray(sliced_image['image']) for sliced_image in sliced_images]
    # big_image = concatenate_images(image_list=image_list,image_size=(1334,1334),num_cols=3,num_rows=3,spacing=20)
    # big_image.show()


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



