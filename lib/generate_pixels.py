import glob
import pandas as pd
import numpy as np
from keras.preprocessing import image

X = []
for filename in sorted(glob.glob("/home/lingyi/Desktop/repo/Fall2019-proj3-sec1--proj3-sec1-grp3/data/test/*.jpg")):
  img = image.load_img(filename, color_mode = "rgb", target_size=(75, 100, 3))
  x = image.img_to_array(img, dtype=int)
  list1 = x.flatten()
  str1 = ' '.join(str(e) for e in list1)
  X.append(str1)
  
  
df = pd.read_csv('/home/lingyi/Desktop/repo/Fall2019-proj3-sec1--proj3-sec1-grp3/data/label_test.csv')
Y = df["emotion_idx"]
data = {'emotion':Y, 'pixels':X} 
  
# Create DataFrame 
df_pixels = pd.DataFrame(data) 
df_pixels.to_csv("/home/lingyi/Desktop/repo/Fall2019-proj3-sec1--proj3-sec1-grp3/dataset_rgb_aug_test_250.csv")