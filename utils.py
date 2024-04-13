# -*- coding: utf-8 -*-
import sys
sys.path.append(r'D:\project\LitchiDetection\sahi-main\sahi')
import streamlit as st
from PIL import Image


from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from predict_res.slice_predict import concatenate_images


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
                result, slice_res = get_sliced_prediction(
                    uploaded_image,
                    model,
                    slice_height=1334,
                    slice_width=1334,
                    overlap_height_ratio=0.25,
                    overlap_width_ratio=0,
                    perform_standard_pred=False,
                    get_slice_result=True,
                )
                #预测切片大图
                sliced_images = [sli_res.export_visuals() for sli_res in slice_res]
                image_list = [Image.fromarray(sliced_image['image']) for sliced_image in sliced_images]
                big_image = concatenate_images(image_list=image_list, image_size=(1334, 1334), num_cols=3, num_rows=3,
                                               spacing=20)
                #预测结果大图
                res_plotted = result.export_visuals()['image']
                res_times = result.export_visuals()['elapsed_time']
                res_class_num = result.count_predictions_by_class() #统计各个类别的数目,返回一个字典
                num_leaves =  'Fade leaves: ' +  str(res_class_num[0]) + '  Normal leaves: ' + str(res_class_num[1])
                fade_degree = 'Fade degree : ' + f"{res_class_num[0]/(res_class_num[0]+res_class_num[1]):.4f}"
                exc_time = f"Execution Time: {res_times:.2f} seconds"
                try:
                    st_count.write(num_leaves + '\n\n' + fade_degree + '\n\n' + exc_time)
                    st_frame.image(big_image,
                                   caption="Sliced Image",
                                   use_column_width=True,
                                   width=1200,
                                   )
                    st_image = st.empty()
                    st_image.image(res_plotted,
                                   caption="Detected Image",
                                   use_column_width=True,
                                   width=1200,
                                   )
                except Exception as ex:
                    st.write("No image is uploaded yet!")
                    st.write(ex)


