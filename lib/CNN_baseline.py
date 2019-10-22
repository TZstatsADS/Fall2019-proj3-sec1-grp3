# https://www.kaggle.com/shawon10/facial-expression-detection-cnn
from google.colab import drive
drive.mount('/content/gdrive/')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("./gdrive/My Drive/Semesters/19 Fall/Applied data science/Project3/data/input"))

import tensorflow as tf


import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filname = './gdrive/My Drive/Semesters/19 Fall/Applied data science/Project3/data/dataset_rgb_aug_train_45000.csv'
label_map = ['Neutral', 'Happy', 'Sad', 'Angry', 'Surprised', 'Disgusted', 'Fearful', "Happily surprised", "Happily disgusted", 
               "Sadly angry", "Angrily disgusted", "Appalled", "Hatred", "Angrily surprised", "Sadly surprised", "Disgustedly surprised", 
              "Fearfully surprised", "Awed", "Sadly fearful", "Fearfully disgusted", "Fearfully angry", "Sadly disgusted"]
# names=['emotion','pixels','usage']
df=pd.read_csv('./gdrive/My Drive/Semesters/19 Fall/Applied data science/Project3/data/dataset_rgb_aug_train_45000.csv', na_filter=False, index_col=None)
im=df['pixels']
df.head(10)




'''get data'''
def getData(filname):
    # images are 75*100
    # N = 2500
    Y = []
    X = []
    first = True
    i = 0
    for line in open(filname):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[1])-1)
            X.append([int(p) for p in row[2].split()])
            i += 1
            if i == 20000:
              break

    X, Y = np.array(X) / 255.0, np.array(Y)
    return X, Y[0:20000]
  
  
X, Y = getData(filname)
num_class = len(set(Y))
print(num_class)

# keras with tensorflow backend
N, D = X.shape
X = X.reshape(N, 75, 100, 3)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)



'''build CNN model'''

from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization

def my_model():
    model = Sequential()
    input_shape = (75,100,3)
    model.add(Conv2D(64, (5, 5), input_shape=input_shape,activation='relu', padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(22))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')
    # UNCOMMENT THIS TO VIEW THE ARCHITECTURE
    #model.summary()
    
    return model
model=my_model()
model.summary()


'''train model'''
def my_model():
    model = Sequential()
    input_shape = (75,100,3)
    model.add(Conv2D(64, (5, 5), input_shape=input_shape,activation='relu', padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(22))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')
    # UNCOMMENT THIS TO VIEW THE ARCHITECTURE
    #model.summary()
    
    return model
model=my_model()
model.summary()



path_model='./gdrive/My Drive/Semesters/19 Fall/Applied data science/Project3/output/model_filter_own_data_rgb_aug_train.h5' # save model at this location after each epoch
K.tensorflow_backend.clear_session() # destroys the current graph and builds a new one
model=my_model() # create the model
K.set_value(model.optimizer.lr,1e-3) # set the learning rate
# fit the model
h=model.fit(x=X_train,     
            y=y_train, 
            batch_size=64, 
            epochs=20, 
            verbose=1, 
            validation_data=(X_test,y_test),
            shuffle=True,
            callbacks=[
                ModelCheckpoint(filepath=path_model),
            ]
            )



'''compute the accuracy'''

from keras.models import load_model
model = load_model("./gdrive/My Drive/Semesters/19 Fall/Applied data science/Project3/output/model_filter_own_data_rgb_aug_train.h5")
filname = './gdrive/My Drive/Semesters/19 Fall/Applied data science/Project3/data/dataset_rgb_aug_test_250.csv'
label_map = ['Neutral', 'Happy', 'Sad', 'Angry', 'Surprised', 'Disgusted', 'Fearful', "Happily surprised", "Happily disgusted", 
               "Sadly angry", "Angrily disgusted", "Appalled", "Hatred", "Angrily surprised", "Sadly surprised", "Disgustedly surprised", 
              "Fearfully surprised", "Awed", "Sadly fearful", "Fearfully disgusted", "Fearfully angry", "Sadly disgusted"]
# names=['emotion','pixels','usage']
df=pd.read_csv('./gdrive/My Drive/Semesters/19 Fall/Applied data science/Project3/data/dataset_rgb_aug_test_250.csv', na_filter=False, index_col=None)
im=df['pixels']
df.head(10)

def getData(filname):
    # images are 75*100
    # N = 2500
    Y = []
    X = []
    first = True
    i = 0
    for line in open(filname):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[1])-1)
            X.append([int(p) for p in row[2].split()])
            i += 1

    X, Y = np.array(X) / 255.0, np.array(Y)
    return X, Y
  
  
X, Y = getData(filname)
num_class = len(set(Y))
print(num_class)

# keras with tensorflow backend
N, D = X.shape
X = X.reshape(N, 75, 100, 3)


X_test = X
y_test = (np.arange(num_class) == Y[:, None]).astype(np.float32)

from keras import metrics
scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


