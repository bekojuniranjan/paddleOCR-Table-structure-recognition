import os
import cv2
from paddleocr import PPStructure,draw_structure_result,save_structure_res
import numpy as np

table_engine = PPStructure(show_log=False, layout=False )

save_folder = './output'
img_path = 'table_image/1.png'
img = cv2.imread(img_path)
result = table_engine(img)
print(np.array(result[0]['res']['html']))
