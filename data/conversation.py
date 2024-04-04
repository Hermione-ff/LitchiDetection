import os
import json
import shutil
import random

# 定义要处理的文件夹路径
folder_path = r"E:\毕业论文\fan_153张树叶"
train_dir = r"D:\project\LitchiDetection\data\dataset\Litchi\train"
val_dir = r"D:\project\LitchiDetection\data\dataset\Litchi\val"
test_dir = r"D:\project\LitchiDetection\data\dataset\Litchi\test"
photo_type = 'jpeg'

# 类别映射，将文本标签映射为整数类别
class_mapping = {
    "fade_leaf": 0, "leaf": 1
}

# 获取子文件夹路径
def get_subfolders(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    return subfolders

# 清理数据
def clean_data(folder_path):
    subfolders = get_subfolders(folder_path)
    for folder_path_ in subfolders:
        for root, dirs, files in os.walk(folder_path_):
            for file in files:
                # 检查文件是否是图片文件
                if file.endswith("." + photo_type):
                    jpg_path = os.path.join(root, file)
                    json_path = os.path.splitext(jpg_path)[0] + ".json"
                    # 检查对应的json文件是否存在
                    if not os.path.exists(json_path):
                        # 如果不存在则删除图片文件
                        print("remove {}".format(jpg_path))
                        os.remove(jpg_path)
                    else:
                        # 读取json文件内容
                        with open(json_path, "r") as f:
                            data = json.load(f)
                        # 检查shapes是否为空
                        if not data["shapes"]:
                            # 如果为空，则删除文件
                            os.remove(jpg_path)
                            os.remove(json_path)
                elif file.endswith(".json"):
                    json_path = os.path.join(root, file)
                    jpg_path = os.path.splitext(json_path)[0] + "." + photo_type
                    # 检查对应的jpg文件是否存在
                    if not os.path.exists(jpg_path):
                        # 如果不存在则删除json文件
                        print("remove {}".format(jpg_path))
                        os.remove(json_path)
        print('文件夹{}已完成'.format(os.path.basename(folder_path_)))

# 转换 JSON 到 YOLO
def json_to_yolo(json_path, output_path, class_mapping):
    with open(json_path, 'r') as f:
        data = json.load(f)
    image_width = data["imageWidth"]
    image_height = data["imageHeight"]
    with open(output_path, 'w') as f:
        for shape in data["shapes"]:
            label = shape["label"]
            if label not in class_mapping:
                continue  # 跳过不在类别映射中的目标
            class_id = class_mapping[label]
            points = shape["points"]
            x_min = min(points, key=lambda x: x[0])[0]
            x_max = max(points, key=lambda x: x[0])[0]
            y_min = min(points, key=lambda x: x[1])[1]
            y_max = max(points, key=lambda x: x[1])[1]
            x_center = (x_min + x_max) / 2 / image_width
            y_center = (y_min + y_max) / 2 / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# 分割数据集为训练集、验证集和测试集
def split_dataset(images_list, labels_list):
    random.shuffle(images_list)
    num_images = len(images_list)
    num_train = int(num_images * 0.7)
    num_val = int(num_images * 0.2)
    num_test = num_images - num_train - num_val

    train_images = images_list[:num_train]
    val_images = images_list[num_train:num_train + num_val]
    test_images = images_list[num_train + num_val:]
    train_labels = [os.path.splitext(image)[0] + ".txt" for image in train_images]
    val_labels = [os.path.splitext(image)[0] + ".txt" for image in val_images]
    test_labels = [os.path.splitext(image)[0] + ".txt" for image in test_images]

    return train_images, val_images, test_images, train_labels, val_labels, test_labels

# 创建目标文件夹
def create_folders(train_dir, val_dir, test_dir):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "labels"), exist_ok=True)

# 将图片和标注文件复制到相应文件夹
def copy_files(images_list, labels_list, images_dir, labels_dir):
    for image_path, label_path in zip(images_list, labels_list):
        if "无任务" in image_path or "无任务" in label_path:
            continue
        shutil.copy(image_path, os.path.join(images_dir, os.path.basename(image_path)))
        shutil.copy(label_path, os.path.join(labels_dir, os.path.basename(label_path)))


if __name__ == '__main__':

    # 清理数据
    clean_data(folder_path)

    # 转换 JSON 到 YOLO
    for folder_path_ in get_subfolders(folder_path):
        for root, _, files in os.walk(folder_path_):
            for file in files:
                if file.endswith(".json"):
                    json_path = os.path.join(root, file)
                    txt_path = os.path.splitext(json_path)[0] + ".txt"
                    json_to_yolo(json_path, txt_path, class_mapping)

    print("JSON TO YOLO 转换完成")

    # 获取所有图片和标注文件的路径
    images_list = []
    labels_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith("." + photo_type):
                images_list.append(os.path.join(root, file))
            elif file.endswith(".txt"):
                labels_list.append(os.path.join(root, file))

    # 分割数据集
    train_images, val_images, test_images, train_labels, val_labels, test_labels = split_dataset(images_list, labels_list)

    # 创建目标文件夹
    create_folders(train_dir, val_dir, test_dir)

    # 将图片和标注文件移动到相应文件夹
    copy_files(train_images, train_labels, os.path.join(train_dir, "images"), os.path.join(train_dir, "labels"))
    copy_files(val_images, val_labels, os.path.join(val_dir, "images"), os.path.join(val_dir, "labels"))
    copy_files(test_images, test_labels, os.path.join(test_dir, "images"), os.path.join(test_dir, "labels"))

    print("转换并分割数据集完成")
