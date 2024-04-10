import os
import json

def labelme_to_coco(labelme_json_folder, save_path):
    # 创建 COCO 格式的字典
    coco_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 类别名称和 ID 映射
    category_id_map = {}
    category_id = 1

    # 读取 LabelMe 标注文件夹中的每个 JSON 文件
    for filename in os.listdir(labelme_json_folder):
        if filename.endswith('.json'):
            with open(os.path.join(labelme_json_folder, filename), 'r') as f:
                labelme_data = json.load(f)

            # 添加图像信息
            image_info = {
                "id": len(coco_data["images"]) + 1,
                "file_name": labelme_data["imagePath"],
                "width": labelme_data["imageWidth"],
                "height": labelme_data["imageHeight"]
            }
            coco_data["images"].append(image_info)

            # 添加标注信息
            for shape in labelme_data["shapes"]:
                category = shape["label"]
                if category not in category_id_map:
                    category_id_map[category] = category_id
                    category_id += 1

                category_id_label = category_id_map[category]

                bbox_x = min(point[0] for point in shape["points"])
                bbox_y = min(point[1] for point in shape["points"])
                bbox_width = max(point[0] for point in shape["points"]) - bbox_x
                bbox_height = max(point[1] for point in shape["points"]) - bbox_y

                annotation = {
                    "id": len(coco_data["annotations"]) + 1,
                    "image_id": image_info["id"],
                    "category_id": category_id_label,
                    "bbox": [bbox_x, bbox_y, bbox_width, bbox_height],
                    "iscrowd": 0,
                    "area": bbox_width * bbox_height,
                    "ignore": 0
                }
                coco_data["annotations"].append(annotation)

    # 添加类别信息
    for category, category_id in category_id_map.items():
        coco_data["categories"].append({
            "id": category_id,
            "name": category,
            "supercategory": "object"
        })

    # 保存为 JSON 文件
    with open(save_path, 'w') as f:
        json.dump(coco_data, f)

if __name__ == '__main__':
    # # 使用示例
    labelme_json_folder = r"E:\毕业论文\leaves_labels\val"
    save_path = r"E:\毕业论文\leaves_labels\valcoco.json"
    labelme_to_coco(labelme_json_folder, save_path)
