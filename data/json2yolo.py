import os
import json
import shutil

# 定义要处理的文件夹路径
folder_path = r"E:\毕业论文\leaves"
train_dir = r"D:\project\LitchiDetection\data\dataset\Litchi\train"
val_dir = r"D:\project\LitchiDetection\data\dataset\Litchi\val"
photo_type = 'jpeg'

# 类别映射，将文本标签映射为整数类别
class_mapping = {
    "fade_leaf": 0, "leaf": 1
}

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

# 创建目标文件夹
def create_folders(train_dir, val_dir):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)

# 复制文件到相应文件夹
def copy_files(images_list, labels_list, images_dir, labels_dir):
    for image_path, label_path in zip(images_list, labels_list):
        shutil.copy(image_path, os.path.join(images_dir, os.path.basename(image_path)))
        shutil.copy(label_path, os.path.join(labels_dir, os.path.basename(label_path)))

# 获取所有图片和标注文件的路径
def get_image_and_label_paths(folder_path):
    image_paths = []
    label_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                image_path = os.path.splitext(json_path)[0] + f'.{photo_type}'
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    label_paths.append(json_path)
    return image_paths, label_paths



if __name__ == '__main__':
    # 获取训练集和验证集的图片和标注文件路径
    train_image_paths, train_label_paths = get_image_and_label_paths(os.path.join(folder_path, 'train'))
    val_image_paths, val_label_paths = get_image_and_label_paths(os.path.join(folder_path, 'val'))

    # 创建目标文件夹
    create_folders(train_dir, val_dir)

    # 复制文件到训练集和验证集文件夹中
    copy_files(train_image_paths, train_label_paths, os.path.join(train_dir, 'images'), os.path.join(train_dir, 'labels'))
    copy_files(val_image_paths, val_label_paths, os.path.join(val_dir, 'images'), os.path.join(val_dir, 'labels'))

    # 删除标签文件夹中的json文件
    for folder in [os.path.join(train_dir, 'labels'), os.path.join(val_dir, 'labels')]:
        for file_name in os.listdir(folder):
            if file_name.endswith('.json'):
                os.remove(os.path.join(folder, file_name))

    # 转换训练集和验证集的标注文件为YOLO格式
    for label_path in train_label_paths:
        output_path = os.path.join(train_dir, 'labels', os.path.basename(os.path.splitext(label_path)[0] + '.txt'))
        json_to_yolo(label_path, output_path, class_mapping)

    for label_path in val_label_paths:
        output_path = os.path.join(val_dir, 'labels', os.path.basename(os.path.splitext(label_path)[0] + '.txt'))
        json_to_yolo(label_path, output_path, class_mapping)

    print("Conversion completed successfully.")
