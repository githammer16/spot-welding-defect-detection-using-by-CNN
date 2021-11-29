# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 22:33:17 2021

@author: ilhan
"""

#https://www.kaggle.com/gpreda/cats-or-dogs-using-cnn-with-transfer-learning
import pydot
import graphviz
import tensorflow.compat.v2 as tf
import tensorflow.compat.v2 as tf
from tensorflow.python import tf2
import numpy as np
import sys
import tensorflow
import os, cv2, random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from random import shuffle 
from IPython.display import SVG

import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator

import tensorflow.keras as keras

from tensorflow.keras.utils import plot_model

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

from tensorflow.keras.utils import plot_model

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from keras.utils import plot_model

from tensorflow.keras.utils import plot_model

#matplotlib inline 

TEST_SIZE = 0.5
RANDOM_STATE = 42
BATCH_SIZE = 20
NO_EPOCHS = 3
NUM_CLASSES = 2
SAMPLE_SIZE = 650
PATH = '/resnet50_punta/'
TRAIN_FOLDER = '../VGG16_punta/working/train/'
TEST_FOLDER =  '../VGG16_punta/working/test/'
IMG_SIZE = 224
RESNET_WEIGHTS_PATH = '../resnet50_punta/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



weights_path = '../resnet50_punta/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
#ResNet50 = keras.applications.resnet50(weights=weights_path ,include_top=False, input_shape=input_shape)

from tensorflow.keras.applications.resnet import ResNet50
model = ResNet50(weights='imagenet')
model_notop = ResNet50(weights='imagenet', include_top=False)


train_image_list = os.listdir("../VGG16_punta/working/train/")[0:SAMPLE_SIZE]
test_image_list = os.listdir("../VGG16_punta/working/test/")

def label_pet_image_one_hot_encoder(img):
    pet = img.split('.')[-3]
    if pet == 'Puntanok': return [1,0]
    elif pet == 'Okpunta': return [0,1]

def process_data(data_image_list, DATA_FOLDER, isTrain=True):
    data_df = []
    for img in tqdm(data_image_list):
        path = os.path.join(DATA_FOLDER,img)
        if(isTrain):
            label = label_pet_image_one_hot_encoder(img)
        else:
            label = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        data_df.append([np.array(img),np.array(label)])
    shuffle(data_df)
    return data_df

def plot_image_list_count(data_image_list):
    labels = []
    for img in data_image_list:
        labels.append(img.split('.')[-3])
    sns.countplot(labels)
    plt.title('Okpunta ve Puntanok')
    
plot_image_list_count(train_image_list)

plot_image_list_count(os.listdir(TRAIN_FOLDER))

train = process_data(train_image_list, TRAIN_FOLDER)

def show_images(data, isTest=False):
    f, ax = plt.subplots(5,5, figsize=(15,15))
    for i,data in enumerate(data[:25]):
        img_num = data[1]
        img_data = data[0]
        label = np.argmax(img_num)
        if label  == 1: 
            str_label='Okpunta'
        elif label == 0: 
            str_label='Puntanok'
        if(isTest):
            str_label="None"
        ax[i//5, i%5].imshow(img_data)
        ax[i//5, i%5].axis('off')
        ax[i//5, i%5].set_title("Label: {}".format(str_label))
    plt.show()

show_images(train)

test = process_data(test_image_list, TEST_FOLDER, False)

show_images(test,True)

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
y = np.array([i[1] for i in train])



model = Sequential()
model.add(ResNet50(include_top=False, pooling='max', weights='imagenet'))
model.add(Dense(NUM_CLASSES, activation='softmax'))
# ResNet-50 model is already trained, should not be trained
model.layers[0].trainable = True

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#plot_model(model, to_file='model.png')
#SVG(model_to_dot(model).create(prog='dot', format='svg'))


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)








train_model = model.fit(X_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=NO_EPOCHS,
                  verbose=1,
                  validation_data=(X_val, y_val))

def plot_accuracy_and_loss(train_model):
    hist = train_model.history
    acc = hist['acc']
    val_acc = hist['val_acc']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = range(len(acc))
    f, ax = plt.subplots(1,2, figsize=(14,6))
    ax[0].plot(epochs, acc, 'g', label='Training accuracy')
    ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
    ax[0].set_title('Training and validation accuracy')
    ax[0].legend()
    ax[1].plot(epochs, loss, 'g', label='Training loss')
    ax[1].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[1].set_title('Training and validation loss')
    ax[1].legend()
    plt.show()
plot_accuracy_and_loss(train_model)

# fine-tune the model
checkpoint_path = 'model_0_vgg16.h5'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

score = model.evaluate(X_val, y_val, verbose=0,callbacks=[model_checkpoint_callback])

print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

#get the predictions for the test data
predicted_classes = model.predict_classes(X_val)
#get the indices to be plotted
y_true = np.argmax(y_val,axis=1)

correct = np.nonzero(predicted_classes==y_true)[0]
incorrect = np.nonzero(predicted_classes!=y_true)[0]

target_names = ["Class {}:".format(i) for i in range(NUM_CLASSES)]
print(classification_report(y_true, predicted_classes, target_names=target_names))