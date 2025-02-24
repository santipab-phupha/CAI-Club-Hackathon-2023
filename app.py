import numpy as np
from PIL import Image
import PIL.Image as Image
import csv
from streamlit_echarts import st_echarts
from st_on_hover_tabs import on_hover_tabs
import streamlit as st
st.set_page_config(layout="wide")
from streamlit_drawable_canvas import st_canvas
from transformers import AutoFeatureExtractor, SwinForImageClassification
import warnings
from torchvision import transforms
from datasets import load_dataset
import cv2
import torch
from torch import nn
from typing import List, Callable, Optional
import os
import pandas as pd
import pydicom
import openai
from openai import OpenAI
from IPython.display import Image, display
import responses
from PIL import Image
import requests
from io import BytesIO
import io
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt



st.markdown('''
<style>
    section[data-testid='stSidebar'] {
        background-color: #111;
        min-width: unset !important;
        width: unset !important;
        flex-shrink: unset !important;
    }

    button[kind="header"] {
        background-color: transparent;
        color: rgb(180, 167, 141);
    }

    @media (hover) {
        /* header element to be removed */
        header["data"-testid="stHeader"] {
            display: none;
        }

        /* The navigation menu specs and size */
        section[data-testid='stSidebar'] > div {
            height: 100%;
            width: 95px;
            position: relative;
            z-index: 1;
            top: 0;
            left: 0;
            background-color: #111;
            overflow-x: hidden;
            transition: 0.5s ease;
            padding-top: 60px;
            white-space: nowrap;
        }

        /* The navigation menu open and close on hover and size */
        /* section[data-testid='stSidebar'] > div {
        height: 100%;
        width: 75px; /* Put some width to hover on. */
        /* } 

        /* ON HOVER */
        section[data-testid='stSidebar'] > div:hover{
        width: 300px;
        }

        /* The button on the streamlit navigation menu - hidden */
        button[kind="header"] {
            display: none;
        }
    }

    @media (max-width: 272px) {
        section["data"-testid='stSidebar'] > div {
            width: 15rem;
        }/.
    }
</style>
''', unsafe_allow_html=True)

# Define CSS styling for centering
centered_style = """
        display: flex;
        justify-content: center;
"""

st.markdown(
    """
<div style='border: 2px solid blue; border-radius: 5px; padding: 10px; background-color: rgba(255, 255, 255, 0.5);'>
    <h1 style='text-align: center; color: black;'>
    LYSEN
    </h1>
</div>
    """, unsafe_allow_html=True)

with open("assets/css/style.css") as f:
    st.markdown(f"<style> {f.read()} </style>",unsafe_allow_html=True)
with open("assets/webfonts/font.txt") as f:
    st.markdown(f.read(),unsafe_allow_html=True)
# end def

with st.sidebar:
    tabs = on_hover_tabs(tabName=['Home','Detection','Search','Security',], 
    iconName=['🏠','❔','🔍','⚠️'], 
    styles={'navtab': {'background-color': '#111', 'color': '#818181', 'font-size': '18px', 
                    'transition': '.3s', 'white-space': 'nowrap', 'text-transform': 'uppercase'}, 
                    'tabOptionsStyle': 
                    {':hover :hover': {'color': 'red', 'cursor': 'pointer'}}, 'iconStyle': 
                    {'position': 'fixed', 'left': '7.5px', 'text-align': 'left'}, 'tabStyle': 
                    {'list-style-type': 'none', 'margin-bottom': '30px', 'padding-left': '30px'}}, 
                    key="1",default_choice=0)
    st.markdown(
    """
        <div style='border: 2px solid green; padding: 10px; white; margin-top: 5px; margin-buttom: 5px; margin-right: 20px; bottom: 50;'>
            <h1 style='text-align: center; color: orange; font-size: 100%'> 💻 CAI HACKATHON 🌎 </h1>
            <h1 style='text-align: center; color: while; font-size: 100%'> 2023 </h1>
            <h1 style='text-align: center; color: orange; font-size: 100%'> ✨🪨 Spark the Pebbles 🪨✨ </h1>
        </div>
    """, unsafe_allow_html=True)

data_base = []
if tabs == 'Home':
    st.image('home.jpg',use_column_width=True)

if tabs == 'Detection':
    st.markdown(
        """
    <div style=' border-radius: 5px; padding: 5px; background-color: rgba(255, 255, 255, 0.5);'>
        <h3 style='text-align: center; color: white; font-size: 150%'> 🔎 Check Real or Fake Signature 🔍 </h3>
    </div>
        """, unsafe_allow_html=True)
    st.markdown(" ")
    # model_path = "C://Users//santi//Desktop//main_model.h5"
    # model = load_model(model_path)

    # # Define the paths to the test images
    # test_image1_path = 'S__11067395.jpg'
    # test_image2_path = "S__11075586.jpg"

    # # Load and preprocess the test images
    # test_image1 = Image.open(test_image1_path)
    # test_image1 = test_image1.resize((112, 112))
    # test_image1 = img_to_array(test_image1)
    # test_image1 = np.expand_dims(test_image1, axis=0)
    # test_image1 = test_image1.astype('float32')

    # test_image2 = Image.open(test_image2_path)
    # test_image2 = test_image2.resize((112, 112))
    # test_image2 = img_to_array(test_image2)
    # test_image2 = np.expand_dims(test_image2, axis=0)
    # test_image2 = test_image2.astype('float32')

    # # Perform inference on the test images
    # prediction = model.predict([test_image1, test_image2])

    # # Print the similarity score
    # similarity_score = prediction[0][0]
    # print('Similarity Score:', similarity_score)

    # # Display the test images with the similarity score as legend
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(test_image1[0].astype('uint8'))
    # plt.title('Test Image 1')
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(test_image2[0].astype('uint8'))
    # plt.title('Test Image 2')
    # plt.axis('off')

    # plt.suptitle(f'Similarity Score: {similarity_score}', fontsize=12)
    # plt.tight_layout()
    # plt.show()

if tabs == "Search":
    uploaded_files = st.file_uploader(" x", 
        type=["jpg", "jpeg", "png", "dcm"], accept_multiple_files=True)

    if uploaded_files is not None:
        processor = AutoFeatureExtractor.from_pretrained('Santipab/CAI-Club-Hackathon-2023-Santipab')
        model = SwinForImageClassification.from_pretrained('Santipab/CAI-Club-Hackathon-2023-Santipab')
        answer = []
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            img = Image.open(io.BytesIO(file_bytes))
            img_out = img.resize((224,224))
            img_out = np.array(img_out)
            image = img.convert('RGB')
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            # model predicts one of the 1000 ImageNet classes
            predicted_class_idx = logits.argmax(-1).item()
            print("Predicted class:", model.config.id2label[predicted_class_idx])
            answer.append(model.config.id2label[predicted_class_idx])
            st.markdown(
                """
            <div style='border: 2px solid #00CC00; border-radius: 5px; padding: 5px; background-color: #99FF99;'>
                <h3 style='text-align: center; color: #000000; font-size: 180%'> 🔎 The signature from : 🔍 </h3>
            </div>
                """, unsafe_allow_html=True) 
            st.markdown(" ")
            st.markdown(
                    f"""
                    <div style='border: 2px solid #994C00; border-radius: 5px; padding: 5px; background-color: white;'>
                        <h3 style='text-align: center; color: #FF8000; font-size: 180%'> User : {answer[0]} </h3>
                    </div>
                    """, unsafe_allow_html=True)
if tabs == 'create signature':
    st.markdown(
        """
    <div style=' border-radius: 5px; padding: 5px; background-color: rgba(255, 255, 255, 0.5);'>
        <h3 style='text-align: center; color: green; font-size: 180%'> 🧑‍💻 Generate Your Own Signature ✍️ </h3>
    </div>
        """, unsafe_allow_html=True)
    st.markdown(" ")

    
    client = OpenAI(api_key = '')
    response = client.images.generate(
    model="dall-e-3",
    prompt="anime cute girl likes to girl in the anime Oni Chi Chi",
    size="1024x1024",
    quality="standard",
    n=1,
    )
    st.markdown(" ")
    col1, col2, col3 = st.columns(3)
    image_url = response.data[0].url
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    col2.image(img)

