# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 12:25:35 2021

@author: ilhan
"""

from keras.applications import ResNet152
import h5py
import matplotlib
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow import keras
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from zipfile import ZipFile
from PIL import Image
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.applications.resnet import ResNet50, ResNet152, preprocess_input


#print("TensorFlow version:", tf.__version__)

#!unzip -q '../kaggle/input/train.zip'
#!unzip -q '../kaggle/input/test1.zip'


#x = np.zeros((0,), int)

from keras import layers, applications, optimizers, callbacks
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation,GlobalMaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

image_size = 64

input_shape = (image_size, image_size, 3)

epochs = 100
batch_size = 20

#pre_trained_model = ResNet152(input_shape=input_shape, include_top=False, weights='imagenet')


#RESNET_WEIGHTS_PATH = '../resnet152_weights_tf.h5'

model = ResNet152(weights='imagenet')

weights_path = '../resnet152_punta/resnet152_weights_tf.h5'

pre_trained_model = ResNet152(input_shape = (image_size, image_size, 3), 
                                include_top = False, 
                                weights = None)


model.load_weights(weights_path, by_name=True)

#DenseNet152(input_shape=input_shape, include_top=False, weights="imagenet")
    
for layer in pre_trained_model.layers[:5]:
    layer.trainable = False

model = Sequential()
    
last_layer = pre_trained_model.get_layer('conv5_block3_out')
last_output = last_layer.output
    
# Flatten the output layer to 1 dimension
x = GlobalMaxPooling2D()(last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x = Dense(512, activation='relu')(x)
# Add a dropout rate of 0.5
x = Dropout(0.5)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(2, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

opt = tf.keras.optimizers.RMSprop(learning_rate=0.0002, rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    name="RMSprop")

model.compile(loss='binary_crossentropy',
              optimizer = opt,
              #optimizer=optimizers.Adam(lr=1e-6),
              metrics=['accuracy'])

model.summary()

filenames = os.listdir("../VGG16_punta/working/train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'Okpunta':
        categories.append(str(1))
 
    if category == 'Puntanok':
        categories.append(str(0))

df = pd.DataFrame({
    'filename': filenames,
    'category': categories,
    
     
})


df.tail()

#df['category'] = df['category'].astype(str)

df['category'].value_counts().plot.bar()



sample = random.choice(filenames)
image = load_img("../VGG16_punta/working/train/"+sample)
plt.imshow(image)
plt.show()


train_df, validate_df = train_test_split(df, test_size=0.1)
train_df = train_df.reset_index()
validate_df = validate_df.reset_index()

# validate_df = validate_df.sample(n=100).reset_index() # use for fast testing code purpose
# train_df = train_df.sample(n=1800).reset_index() # use for fast testing code purpose

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

train_datagen = ImageDataGenerator(
    rotation_range=16,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "../VGG16_punta/working/train/", 
    x_col='filename',
    y_col='category',
    class_mode='categorical',
    target_size=(image_size, image_size),
    batch_size=batch_size
    
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "../VGG16_punta/working/train/",  
    x_col='filename',
    y_col='category',
    class_mode='categorical',
    target_size=(image_size, image_size),
    batch_size=batch_size
)




example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "../VGG16_punta/working/train/", 
    x_col='filename',
    y_col='category',
    class_mode='categorical'
)

image = load_img("../VGG16_punta/working/train/"+example_df['filename'].values[0])
plt.imshow(image);

plt.figure(figsize=(15, 15))
for i in range(9):
    plt.subplot(3, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()

'''
# fine-tune the model
checkpoint_path = 'model_0_vgg16.h5'

#eski code
#cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                            #save_weights_only=False,
                                            #monitor='val_loss',
                                            #save_best_only=True)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
'''



#model.load_weights(weights_path, by_name=True)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)

model_checkpoint_callback = ModelCheckpoint('**path_you_want_to_save**/best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')


history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=[model_checkpoint_callback])
   

