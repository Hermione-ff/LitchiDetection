import os
import random
import shutil

def creat_datasets(source_folder):
    '''
    将包含有同名的图片和json划分为数据集
    '''
    # 源文件夹和目标文件夹
    train_folder = r"D:\project\LitchiDetection\data\leaves\train"
    val_folder = r"D:\project\LitchiDetection\data\leaves\val"
    test_folder = r"D:\project\LitchiDetection\data\leaves\test"

    # 如果目标文件夹不存在，则创建它们
    for folder in [train_folder, val_folder, test_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # 查找所有匹配的文件
    files = []
    for file_name in os.listdir(source_folder):
        if file_name.endswith(".json"):
            files.append(file_name[:-5])  # 移除文件扩展名后添加到列表中

    # 随机打乱文件列表
    random.shuffle(files)

    # 计算文件数量
    total_files = len(files)
    train_count = int(total_files * 0.7)
    val_count = int(total_files * 0.2)
    test_count = total_files - train_count - val_count

    # 将文件复制到相应的目标文件夹中
    for i, file_name in enumerate(files):
        if i < train_count:
            shutil.copy(os.path.join(source_folder, file_name + ".json"), train_folder)
            shutil.copy(os.path.join(source_folder, file_name + ".jpeg"), train_folder)
        elif i < train_count + val_count:
            shutil.copy(os.path.join(source_folder, file_name + ".json"), val_folder)
            shutil.copy(os.path.join(source_folder, file_name + ".jpeg"), val_folder)
        else:
            shutil.copy(os.path.join(source_folder, file_name + ".json"), test_folder)
            shutil.copy(os.path.join(source_folder, file_name + ".jpeg"), test_folder)

    print("划分数据集完毕")
    return

if __name__ == '__main__':

    #将会在该文件下创建三个文件夹train test val
    source_folder = "D:\project\LitchiDetection\data\leaves"


    creat_datasets(source_folder)