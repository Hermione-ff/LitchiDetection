#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
import streamlit as st

import config
from utils import  infer_uploaded_image, load_slic_model

# 设置页面布局
st.set_page_config(
    page_title="Interactive Interface for YOLOv8",  # 页面标题
    page_icon="🤖", # 页面图标
    layout="wide", # 页面布局
    initial_sidebar_state="expanded" # 初始侧边栏状态
    )

# 主页面标题
st.title("Detection of Fading Degree of Litchi Branches")

# 侧边栏
st.sidebar.header("DL Model Config")

# 模型选项
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection"]
)

model_type = None
if task_type == "Detection":
    model_type = st.sidebar.selectbox(
        "Select Model",
        config.DETECTION_MODEL_LIST   #模型列表
    )
else:
    st.error("Currently only 'Detection' function is implemented")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100  #按键,选择模型置信度

model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
else:
    st.error("Please Select Model in Sidebar")

# 加载预训练的深度学习模型
try:
    slice_model = load_slic_model(model_path,confidence)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

# 图片选项
st.sidebar.header("Image Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)

source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # 图片
    infer_uploaded_image(slice_model,model_type)
else:
    st.error("Currently only 'Image' source are implemented")