# -*- coding: utf-8 -*-
import sys
sys.path.append(r'D:\project\LitchiDetection\sahi-main\sahi')
import streamlit as st
from PIL import Image


from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


@st.cache_resource
def load_slic_model(model_path,conf):
    """
    从指定的model_path加载YOLO切片对象检测模型。
    参数:
        model_path (str): YOLO模型文件的路径。
    返回:
        YOLO切片对象检测模型。
    """
    # 初始化检测模型
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=conf,
        device="cuda:0",  # or 'cuda:0'
    )
    return detection_model


def infer_uploaded_image(model):
    """
    执行上传图片的推断
    :param conf: YOLOv8模型的置信度
    :param model: 包含YOLOv8模型的YOLOv8类的实例。
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    row1 = st.empty()
    with row1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # 将上传的图片添加到页面并附上说明
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True,
                width=1200,
            )

    if source_img:
        if st.button("Execution"):
            row1.empty()
            st_count = st.empty()
            st_frame = st.empty()
            with st.spinner("Running..."):
                # 使用切片检测模型检测图片
                result = get_sliced_prediction(
                    uploaded_image,
                    model,
                    slice_height=1600,
                    slice_width=1600,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                    perform_standard_pred=True,
                )
                res_plotted = result.export_visuals()['image']
                res_times = result.export_visuals()['elapsed_time']
                res_class_num = result.count_predictions_by_class() #统计各个类别的数目,返回一个字典
                num_leaves =  'Fade leaves: ' +  str(res_class_num[0]) + '  Normal leaves: ' + str(res_class_num[1])
                fade_degree = 'Fade degree : ' + f"{res_class_num[0]/(res_class_num[0]+res_class_num[1]):.4f}"
                exc_time = f"Execution Time: {res_times:.2f} seconds"
                try:
                    st_count.write(num_leaves + '\n\n' + fade_degree + '\n\n' + exc_time)
                    st_frame.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True,
                             width=1200,
                                   )
                except Exception as ex:
                    st.write("No image is uploaded yet!")
                    st.write(ex)


