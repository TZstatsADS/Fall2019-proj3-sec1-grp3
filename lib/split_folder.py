import os, os.path, shutil
import glob
import pandas as pd

folder_path = "/home/lingyi/Desktop/repo/Fall2019-proj3-sec1--proj3-sec1-grp3/data"
train_path = "/home/lingyi/Desktop/repo/Fall2019-proj3-sec1--proj3-sec1-grp3/data/train"

images = [f for f in sorted(os.listdir(train_path)) if f.endswith(".jpg")]

test_images = list( images[i] for i in range(0, 2500, 10))

folder_name = "test"
new_path = os.path.join(folder_path, folder_name)
os.makedirs(new_path)
for image in test_images:
    old_image_path = os.path.join(train_path, image)
    new_image_path = os.path.join(new_path, image)
    shutil.move(old_image_path, new_image_path)



df = pd.read_csv('/home/lingyi/Desktop/repo/Fall2019-proj3-sec1--proj3-sec1-grp3/data/label.csv')
df_test = df.loc[range(0, 2500, 10), :]
df_train = df.drop(range(0, 2500, 10))

df_test.to_csv('/home/lingyi/Desktop/repo/Fall2019-proj3-sec1--proj3-sec1-grp3/data/label_test.csv')
df_train.to_csv('/home/lingyi/Desktop/repo/Fall2019-proj3-sec1--proj3-sec1-grp3/data/label_train.csv')
