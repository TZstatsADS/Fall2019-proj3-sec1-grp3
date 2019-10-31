import os
from PIL import Image
import numpy as np
import cv2 as cv


read_path = "./data/data_1"

save_path = './data/data_2'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
import pandas as pd 
data = pd.read_csv("./data/label/label.csv") 
file_list = sorted(os.listdir(read_path))
emotion_idx = data['emotion_idx']



for i in range(len(file_list)):
    label = str(emotion_idx[i])
    subfolder = os.path.join(save_path, label)
    image = file_list[i]
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    img_gray = cv.imread(os.path.join(read_path, image),0)
    img_gray = cv.resize(img_gray, (48, 48))
    cv.imwrite(os.path.join(subfolder, image), img_gray)



