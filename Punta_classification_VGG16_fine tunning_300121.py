# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 14:17:45 2021

@author: ilhan
"""

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


#print("TensorFlow version:", tf.__version__)

#!unzip -q '../kaggle/input/train.zip'
#!unzip -q '../kaggle/input/test1.zip'

filenames = os.listdir("../VGG16_punta/working/train/")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'Puntanok':
        categories.append(str(1))
 
    if category == 'Okpunta':
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

#x = np.zeros((0,), int)

from keras import layers, applications, optimizers, callbacks
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation,GlobalMaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Model

image_size = 64

input_shape = (image_size, image_size, 3)

epochs = 100
batch_size = 20



pre_trained_model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")
    
for layer in pre_trained_model.layers[:5]:
    layer.trainable = False

import visualkeras
    
last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output
    
# Flatten the output layer to 1 dimension
x = GlobalMaxPooling2D()(last_output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='sigmoid')(x)
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

x = Sequential()

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


history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=[model_checkpoint_callback])

loss, accuracy = model.evaluate_generator(validation_generator, total_validate//batch_size, workers=12)
print("Test: accuracy = %f  ;  Test: loss = %f " % (accuracy, loss))

#
import visualkeras


visualkeras.layered_view(model).show()

from PIL import ImageFont

font = ImageFont.truetype("arial.ttf",20,encoding="unic")

visualkeras.layered_view(model,legend=True, font=font).show()

visualkeras.layered_view(model, draw_volume=False).show()



#visualkeras.SpacingDummyLayer()
#model.add(visualkeras.SpacingDummyLayer(spacing=100))
#visualkeras.layered_view(model, spacing=0)

#from tensorflow
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten,Dropout,GlobalMaxPooling2D,ZeroPadding2D
from collections import defaultdict

color_map = defaultdict(dict)
color_map[Conv2D]['fill'] = 'orange'
color_map[ZeroPadding2D]['fill'] = 'gray'
color_map[Dropout]['fill'] = 'pink'
color_map[GlobalMaxPooling2D]['fill'] = 'green'
color_map[Flatten]['fill'] = 'teal'

visualkeras.layered_view(model, color_map= color_map)

visualkeras.layered_view(model, type_ignore= [GlobalMaxPooling2D,Dropout,Flatten])
                         

'''
#print(history.history.keys())
#plt.plot(history.history['accuracy'])  
#plt.plot(history.history['val_accuracy'])
#plt.title('model accuracy')


plt.show()


test_filenames = os.listdir("../puntacnn/working/test/")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "../puntacnn/working/test/", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    batch_size=batch_size,
    target_size=(image_size, image_size),
    shuffle=False
)

import numpy as np
import pandas as pd
import keras
from matplotlib import pyplot as plt
#  "Accuracy"

history_dict = history.history
print(history_dict.keys())



print(history.history.keys())
plt.plot(history.history['accuracy'])  
plt.plot(history.history['val_accuracy'])
#plt.plot(history.history['test_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train acc', 'validation acc'],loc='upper left') 
#plt.legend(['test acc'],loc='upper left') 
plt.show()


plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
#plt.plot(history.history['test loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train loss', 'validation loss'], loc='upper left')  
#plt.legend(['test loss'], loc='upper left')

history_dict = history.history
print(history_dict.keys())



pd.DataFrame(history.history).plot(figsize=(8,5))
#plt.grid(True)
#plt.gca().set_ylim(0, 1)


predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
threshold = 0.5
test_df['category'] = np.where(predict >= threshold, 1,0)

sample_test = test_df.sample(n=9).reset_index()
sample_test.head()
plt.figure(figsize=(12, 12))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("../puntacnn/working/test/"+filename, target_size=(256, 256))
    plt.subplot(3, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')')
plt.tight_layout()
plt.show()

submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)

#plt.figure(figsize=(10,5))
#sns.countplot(submission_df['label'])
plt.title("(Test data)")
'''
