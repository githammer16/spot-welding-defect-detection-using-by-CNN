# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 21:48:45 2021

@author: ilhan
"""


import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import random
import cv2
import os
import gc

from sklearn.model_selection import train_test_split
from keras.applications import InceptionResNetV2
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

print(os.listdir("../VGG16_punta/working"))

train_dir = '../VGG16_punta/working/train'
test_dir = '../VGG16_punta/working/test'

# train_imgs = ['../input/train/{}'.format(i) for i in os.listdir(train_dir)]  #get full data set
train_Puntanok = ['../VGG16_punta/working/train/{}'.format(i) for i in os.listdir(train_dir) if 'Puntanok' in i]  #get dog images
train_Okpunta = ['../VGG16_punta/working/train/{}'.format(i) for i in os.listdir(train_dir) if 'Okpunta' in i]  #get cat images

test_imgs = ['../VGG16_punta/working/test/{}'.format(i) for i in os.listdir(test_dir)] #get test images

size=400
train_imgs = train_Puntanok[0:size] + train_Okpunta[0:size]

random.shuffle(train_imgs)  # shuffle it randomly

img_size = 150

def read_and_process_image(list_of_images):
    """
    Returns three arrays: 
        X is an array of resized images
        y is an array of labels
        l_id an array of Ids for submission
    """
    X = [] # images
    y = [] # labels
    l_id = [] # id for submission
    
    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (img_size,img_size), interpolation=cv2.INTER_CUBIC))  #Read the image
        basename = os.path.basename(image)
        img_num = basename.split('.')[0]
        l_id.append(img_num)
        #get the labels
        if 'Puntanok' in image:
            y.append(1)
        elif 'Okpunta' in image:
            y.append(0)
    
    return X, y, l_id

X, y, l_id = read_and_process_image(train_imgs)



plt.figure(figsize=(20,10))
columns = 5
for i in range(columns):
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.imshow(X[i])
    

X = np.array(X)
y = np.array(y)

sns.countplot(y)
plt.title('Labels for Cats and Dogs')
plt.show()


print("Shape of X_train:",X.shape)
print("Shape of X_val:", y.shape)

    
from sklearn.model_selection import train_test_split
    
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=1)

del X
del y
del train_imgs
del train_Puntanok
del train_Okpunta
gc.collect()

print("Shape of X_train",X_train.shape)
print("Shape of X_val", X_val.shape)


ntrain = len(X_train)
nval = len(X_val)

conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=[150, 150, 3]) 
conv_base.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))   

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])



#model.summary()
batch_size = 20

train_datagen = ImageDataGenerator(rescale=1./255,   #Scale the image between 0 and 1
                                    #rotation_range=30,
                                    #width_shift_range=0.2,
                                    #height_shift_range=0.2,
                                    #shear_range=0.2,
                                    #zoom_range=0.2,
                                    #horizontal_flip=True,
                                    fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255) 

train_generator = train_datagen.flow(X_train, y_train,  batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

epochs = 100

# fine-tune the model
checkpoint_path = 'model_0_vgg16.h5'



model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit_generator(train_generator,
                              steps_per_epoch=ntrain // batch_size,
                              epochs=epochs,
                              validation_data=val_generator,
                              validation_steps=nval // batch_size,
                              callbacks=[model_checkpoint_callback])
                              

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()




'''
X_test, y_test, l_id = read_and_process_image(test_imgs[10:20]) #Y_test in this case will be empty.
x = np.array(X_test)
test_datagen = ImageDataGenerator(rescale=1./255)

i = 0
columns = 5
text_labels = []
plt.figure(figsize=(30,20))
for batch in test_datagen.flow(x, batch_size=1):
    pred = model.predict(batch)
    pred = np.float(pred)
    if pred > 0.5:
        text_labels.append('dog ({:.3f})'.format(pred))
    else:
        text_labels.append('cat ({:.3f})'.format(pred))
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.title('This is a ' + text_labels[i])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % 10 == 0:
        break
plt.show()

'''