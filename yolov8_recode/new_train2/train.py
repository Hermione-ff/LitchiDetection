import os


def train_yolov8():
    # 配置参数
    model_cfg = r'yolov8s.yaml'  # 模型配置文件
    data_cfg = r'/content/drive/MyDrive/LitchiDetection/data/dataset/Litchi.yaml'  # 数据集配置文件
    # pretrained_cfg = r'yolov8m.pt'   不采用预训练
    epochs = '200'  # 训练轮次
    batch_size = '16'  # 批次大小
    imgsz = '640'
    device = '0'  # 使用的设备编号，'0' 表示第一个GPU，'cpu' 表示使用CPU
    patience = '50'  #训练停止阈值
    project = r'/content/drive/MyDrive/LitchiDetection/yolov8_recode' #保存训练结果的项目目录名称
    name = r'new_train4' #用于存储训练日志和输出结果
    cos_lr = 'True'#使用余弦退火学习率调度策略
    save_dir = r'/content/drive/MyDrive/LitchiDetection/yolov8_recode/new_train4'
    optimizer = 'Lion'
    # 构建训练命令
    train_cmd = f'yolo task=detect mode=train model={model_cfg} mode=train data={data_cfg} imgsz={imgsz} ' \
            f'epochs={epochs} batch={batch_size} resume=True device={device} cos_lr={cos_lr} optimizer={optimizer} ' \
            f'patience={patience} project={project} name={name} save_dir={save_dir} '


    # 执行训练命令
    os.system(train_cmd)


if __name__ == '__main__':
    train_yolov8()