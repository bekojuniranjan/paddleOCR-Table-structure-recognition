import streamlit as st
import os
import cv2
from paddleocr import PPStructure,draw_structure_result,save_structure_res
import numpy as np
from PIL import Image
import asyncio

class PaddleTableStructure:
    def __init__(self) -> None:
        self.table_engine = PPStructure(show_log=False, layout=False )
        self.save_folder = './output'
        self.img_path = 'table_image/1.png'

    async def predict_table(self, img_array):
        result = self.table_engine(img_array)
        st.write(result[0]['res']['html'], unsafe_allow_html=True)
    
if __name__ == "__main__":
    st.header("Paddle Table Structure Recognition")
    file = st.file_uploader('Table Image', accept_multiple_files=False, type=['png','jpg'], )
    pts = PaddleTableStructure()
    if file is not None:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        asyncio.run(pts.predict_table(np.array(image)))

