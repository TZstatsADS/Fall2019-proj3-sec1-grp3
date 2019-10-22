
# https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/

import glob
import pandas as pd
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator


aug_times = 4
X = []
for filename in sorted(glob.glob("/home/lingyi/Desktop/repo/Fall2019-proj3-sec1--proj3-sec1-grp3/data/train/*.jpg")):
    img = image.load_img(filename, color_mode = "rgb", target_size=(75, 100, 3))
    data = image.img_to_array(img, dtype = int)
    samples = np.expand_dims(data, 0)    
    datagen = ImageDataGenerator(width_shift_range=[-20,20])
    it = datagen.flow(samples, batch_size=1)
    for i in range(aug_times):
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        x = batch[0].astype('uint8')
        list1 = x.flatten()
        str1 = ' '.join(str(e) for e in list1)
        X.append(str1)


for filename in sorted(glob.glob("/home/lingyi/Desktop/repo/Fall2019-proj3-sec1--proj3-sec1-grp3/data/train/*.jpg")):
    img = image.load_img(filename, color_mode = "rgb", target_size=(75, 100, 3))
    data = image.img_to_array(img, dtype = int)
    samples = np.expand_dims(data, 0)    
    datagen = ImageDataGenerator(height_shift_range=0.15)
    it = datagen.flow(samples, batch_size=1)
    for i in range(aug_times):
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        x = batch[0].astype('uint8')
        list1 = x.flatten()
        str1 = ' '.join(str(e) for e in list1)
        X.append(str1)

for filename in sorted(glob.glob("/home/lingyi/Desktop/repo/Fall2019-proj3-sec1--proj3-sec1-grp3/data/train/*.jpg")):
    img = image.load_img(filename, color_mode = "rgb", target_size=(75, 100, 3))
    data = image.img_to_array(img, dtype = int)
    samples = np.expand_dims(data, 0)    
    datagen = ImageDataGenerator(horizontal_flip=True)
    it = datagen.flow(samples, batch_size=1)
    for i in range(aug_times):
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        x = batch[0].astype('uint8')
        list1 = x.flatten()
        str1 = ' '.join(str(e) for e in list1)
        X.append(str1)

for filename in sorted(glob.glob("/home/lingyi/Desktop/repo/Fall2019-proj3-sec1--proj3-sec1-grp3/data/train/*.jpg")):
    img = image.load_img(filename, color_mode = "rgb", target_size=(75, 100, 3))
    data = image.img_to_array(img, dtype = int)
    samples = np.expand_dims(data, 0)    
    datagen = ImageDataGenerator(brightness_range=[0.3,1.0])
    it = datagen.flow(samples, batch_size=1)
    for i in range(aug_times):
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        x = batch[0].astype('uint8')
        list1 = x.flatten()
        str1 = ' '.join(str(e) for e in list1)
        X.append(str1)

for filename in sorted(glob.glob("/home/lingyi/Desktop/repo/Fall2019-proj3-sec1--proj3-sec1-grp3/data/train/*.jpg")):
    img = image.load_img(filename, color_mode = "rgb", target_size=(75, 100, 3))
    data = image.img_to_array(img, dtype = int)
    samples = np.expand_dims(data, 0)    
    datagen = ImageDataGenerator(zoom_range=[0.5,1])
    it = datagen.flow(samples, batch_size=1)
    for i in range(aug_times):
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        x = batch[0].astype('uint8')
        list1 = x.flatten()
        str1 = ' '.join(str(e) for e in list1)
        X.append(str1)



df = pd.read_csv('/home/lingyi/Desktop/repo/Fall2019-proj3-sec1--proj3-sec1-grp3/data/label_train.csv')   
y = df["emotion_idx"]
Y =  [ele for ele in y for i in range(aug_times)] 
Y = Y*5
data = {'emotion':Y, 'pixels':X} 
  
# Create DataFrame 
df_pixels = pd.DataFrame(data) 
df_pixels.to_csv("/home/lingyi/Desktop/repo/Fall2019-proj3-sec1--proj3-sec1-grp3/dataset_rgb_aug_train_45000.csv")