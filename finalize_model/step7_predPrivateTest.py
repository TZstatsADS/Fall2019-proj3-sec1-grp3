"""
visualize results for test image
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
import cv2 as cv
import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *

read_path = "/home/lingyi/Desktop/finalize_model/data/data_1"
data = pd.read_csv("/home/lingyi/Desktop/finalize_model/data/label/label.csv") 
emotion_idx = data['emotion_idx']
file_list = sorted(os.listdir(read_path))
cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

device = torch.device('cpu')
# net = VGG('VGG19')
# net = ResNet(BasicBlock, [2,2,2,2]) #18
net = ResNet(BasicBlock, [3,4,6,3]) #34

# for i in range(10):

# checkpoint = torch.load('/home/lingyi/Desktop/finalize_model/Project3_VGG19/' + str(i + 1) + '/Test_model.t7', map_location=device)
checkpoint = torch.load('/home/lingyi/Desktop/finalize_model/Project3_VGG19/Project3_2500_Resnet34/Test_model.t7', map_location=device)
net.load_state_dict(checkpoint['net'].state_dict())
# net.cuda()

# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

labels = []
for i in range(len(emotion_idx)):
    raw_img = file_list[i]
    gray = cv.imread(os.path.join(read_path, raw_img),0).astype(np.uint8)

    # gray = rgb2gray(raw_img)
    # gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)

    img = gray[:, :, np.newaxis]

    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)

    net.eval()

    ncrops, c, h, w = np.shape(inputs)

    inputs = inputs.view(-1, c, h, w)
    inputs = Variable(inputs, volatile=True)
    outputs = net(inputs)

    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

    score = F.softmax(outputs_avg)
    _, predicted = torch.max(outputs_avg.data, 0)

    labels.append(int(predicted.cpu().numpy())+1)


true = emotion_idx.tolist()
print(sum(np.array(labels)==np.array(true))/float(len(true)))


