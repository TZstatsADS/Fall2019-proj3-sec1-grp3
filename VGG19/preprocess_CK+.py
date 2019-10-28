# create data and label for data+
#  0=anger 1=disgust, 2=fear, 3=happy, 4=sadness, 5=surprise, 6=contempt
# contain 135,177,75,207,84,249,54 images

import csv
import os
import numpy as np
import h5py
import skimage.io

data_path = '/home/lingyi/Desktop/Project3/all_data/subfolder2250'

c1_path = os.path.join(data_path, '1')
c2_path = os.path.join(data_path, '2')
c3_path = os.path.join(data_path, '3')
c4_path = os.path.join(data_path, '4')
c5_path = os.path.join(data_path, '5')
c6_path = os.path.join(data_path, '6')
c7_path = os.path.join(data_path, '7')
c8_path = os.path.join(data_path, '8')
c9_path = os.path.join(data_path, '9')
c10_path = os.path.join(data_path, '10')
c11_path = os.path.join(data_path, '11')
c12_path = os.path.join(data_path, '12')
c13_path = os.path.join(data_path, '13')
c14_path = os.path.join(data_path, '14')
c15_path = os.path.join(data_path, '15')
c16_path = os.path.join(data_path, '16')
c17_path = os.path.join(data_path, '17')
c18_path = os.path.join(data_path, '18')
c19_path = os.path.join(data_path, '19')
c20_path = os.path.join(data_path, '20')
c21_path = os.path.join(data_path, '21')
c22_path = os.path.join(data_path, '22')

# # Creat the list to store the data and label information
data_x = []
data_y = []

datapath = os.path.join('/home/lingyi/Desktop/Project3/all_data','data_data.h5')
if not os.path.exists(os.path.dirname(datapath)):
    os.makedirs(os.path.dirname(datapath))

# order the file, so the training set will not contain the test set (don't random)
files = os.listdir(c1_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c1_path,filename))
    data_x.append(I.tolist())
    data_y.append(1)

files = os.listdir(c2_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c2_path,filename))
    data_x.append(I.tolist())
    data_y.append(2)

files = os.listdir(c3_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c3_path,filename))
    data_x.append(I.tolist())
    data_y.append(3)

files = os.listdir(c4_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c4_path,filename))
    data_x.append(I.tolist())
    data_y.append(4)

files = os.listdir(c5_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c5_path,filename))
    data_x.append(I.tolist())
    data_y.append(5)

files = os.listdir(c6_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c6_path,filename))
    data_x.append(I.tolist())
    data_y.append(6)

files = os.listdir(c7_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c7_path,filename))
    data_x.append(I.tolist())
    data_y.append(7)

files = os.listdir(c8_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c8_path,filename))
    data_x.append(I.tolist())
    data_y.append(8)

files = os.listdir(c9_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c9_path,filename))
    data_x.append(I.tolist())
    data_y.append(9)

files = os.listdir(c10_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c10_path,filename))
    data_x.append(I.tolist())
    data_y.append(10)

files = os.listdir(c11_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c11_path,filename))
    data_x.append(I.tolist())
    data_y.append(11)

files = os.listdir(c12_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c12_path,filename))
    data_x.append(I.tolist())
    data_y.append(12)

files = os.listdir(c13_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c13_path,filename))
    data_x.append(I.tolist())
    data_y.append(13)

files = os.listdir(c14_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c14_path,filename))
    data_x.append(I.tolist())
    data_y.append(14)

files = os.listdir(c15_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c15_path,filename))
    data_x.append(I.tolist())
    data_y.append(15)

files = os.listdir(c16_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c16_path,filename))
    data_x.append(I.tolist())
    data_y.append(16)

files = os.listdir(c17_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c17_path,filename))
    data_x.append(I.tolist())
    data_y.append(17)

files = os.listdir(c18_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c18_path,filename))
    data_x.append(I.tolist())
    data_y.append(18)

files = os.listdir(c19_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c19_path,filename))
    data_x.append(I.tolist())
    data_y.append(19)

files = os.listdir(c20_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c20_path,filename))
    data_x.append(I.tolist())
    data_y.append(20)

files = os.listdir(c21_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c21_path,filename))
    data_x.append(I.tolist())
    data_y.append(21)

files = os.listdir(c22_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(c22_path,filename))
    data_x.append(I.tolist())
    data_y.append(22)

print(np.shape(data_x))
print(np.shape(data_y))

datafile = h5py.File(datapath, 'w')
datafile.create_dataset("data_pixel", dtype = 'uint8', data=data_x)
datafile.create_dataset("data_label", dtype = 'int64', data=data_y)
datafile.close()

print("Save data finish!!!")
