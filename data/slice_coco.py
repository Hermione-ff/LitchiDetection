# 展示图像
from sahi.utils.file import load_json
from sahi.slicing import slice_coco



# coco图像集地址
image_path = r"E:\毕业论文\leaves_photos\val"
# coco标注文件
coco_annotation_file_path=r"E:\毕业论文\leaves_labels\valcoco.json"
# 加载数据集
coco_dict = load_json(coco_annotation_file_path)



# 保存的coco数据集标注文件名
output_coco_annotation_file_name = "sliced"
# 输出文件夹
output_dir = "valsult"

# 切分数据集
coco_dict, coco_path = slice_coco(
    coco_annotation_file_path=coco_annotation_file_path,
    image_dir=image_path,
    output_coco_annotation_file_name=output_coco_annotation_file_name,
    ignore_negative_samples=False,
    output_dir=output_dir,
    slice_height=1334,
    slice_width=1334,
    overlap_height_ratio=0.25,
    overlap_width_ratio=0,
    min_area_ratio=0.2,
    verbose=False
)

print("切分子图{}张".format(len(coco_dict['images'])))
print("获得标注框{}个".format(len(coco_dict['annotations'])))