# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 18:09:41 2021

@author: ilhan
"""

import numpy as np
import pandas as pd 
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import keras.applications.xception as xception
import zipfile
import sys
import time
import tensorflow.keras as keras

from PIL import Image
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Input
from keras.models import Model, Sequential
from keras.preprocessing import image
#from keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

print('setup successful!')

IMAGE_WIDTH = 224    
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3

# The image net have 1000 categories, and so 1000 outputs, so we get 1000 different features
IMAGE_FEATURES_SIZE = 1000

# For the first run PERFORM_UNZIPPING must be True, but after the first run, 
#it can be set to False skip the unzipping step to save time
PERFORM_UNZIPPING = True 

print('defining constants successful!')

imgs_path = "../VGG16_punta/working/train/"
realtestimgs_path = "../VGG16_punta/working/train/"

def _load_image(img_path):
    img = image.load_img(img_path, target_size = (IMAGE_WIDTH, IMAGE_HEIGHT)) # load the image from the directory
    img = image.img_to_array(img) 
    # add an additional dimension, (i.e. change the shape of each image from (224, 224, 3) to (1, 224, 224, 3)
    # This shape is suitable for training
    img = np.expand_dims(img, axis = 0) 
    # Apply preprocessing for the image, so that the training is faster
    img = xception.preprocess_input(img)
        
    return img

filenames = os.listdir(imgs_path)

categories = []

for filename in filenames:
    category = filename.split('.')[0]
    if category == 'Okpunta':
        categories.append(1)
    if category == 'Puntanok':
        categories.append(str(0))

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

print(df.head())

print('number of elements = ' , len(df))

# see sample image, you can run the same cell again to get a different image
sample = random.choice(filenames)
randomimage = image.load_img(imgs_path + sample)
plt.imshow(randomimage)

# create the xception model used for the feature extraction
model_xception = xception.Xception(include_top = True, input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS),
                       weights = '/makina_ogrenmesi/Xception_punta/xception_weights_tf_dim_ordering_tf_kernels.h5')

# Define call backs
#from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ReduceLROnPlateau
early_stop = EarlyStopping(patience = 7, verbose = 1)

learning_rate_reduction = ReduceLROnPlateau(patience = 3, verbose = 1, factor=0.5)

print('call backs defined!')

def extract_features(model, data_set):
    
    loaded_images = np.zeros((len(data_set), IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
        
    for i, data_set_entry in enumerate (data_set):
        file_name = data_set_entry[0]
        category  = data_set_entry[1]
        path = imgs_path + file_name        
        loaded_img = _load_image(path)
        loaded_images[i, :, :, :] = loaded_img

    preds = model.predict(loaded_images)   
    
    return preds

train_set, validate_set, test_set = np.split(df.sample(frac=1, random_state =43), [int(.82*len(df)), int(.96*len(df))])

# Visualize the data distribution
fig, axis = plt.subplots(1,3)
fig.tight_layout()
fig.set_size_inches(80,20)

ax0 = sns.countplot(data=train_set   , x = 'category', ax = axis[0])
ax1 = sns.countplot(data=validate_set, x = 'category', ax = axis[1])
ax2 = sns.countplot(data=test_set    , x = 'category', ax = axis[2])

ax0.set_title('train data', fontsize = 60)
ax1.set_title('cross validation data', fontsize = 60)
ax2.set_title('test data', fontsize = 60)

ax0.tick_params(labelsize=55)
ax1.tick_params(labelsize=55)
ax2.tick_params(labelsize=55)

print('shape of train_set:   ', np.shape(train_set))
print('shape of validate_set:', np.shape(validate_set))
print('shape of test_set:    ', np.shape(test_set))


# convert the dataframes to numpy matrices
train_set    = train_set.to_numpy()
validate_set = validate_set.to_numpy()
test_set     = test_set.to_numpy()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
# prepare parameters for model.fit
train_x = np.zeros((len(train_set), 1000)) 
train_y = np.zeros((len(train_set)))

validate_x = np.zeros((len(validate_set), 1000)) 
validate_y = np.zeros((len(validate_set) ))

chunk_size = 1000

# extract features from train_set nd save it into train_x
for i, train_set_chunk in enumerate (chunks(train_set, chunk_size)):
    train_x[ (i*chunk_size) : (i*chunk_size + chunk_size)] = extract_features(model_xception, train_set_chunk)    

print('shape of train_x: ',    np.shape(train_x))


# extract features from validate_set nd save it into validate_x
for i, validate_set_chunk in enumerate (chunks(validate_set, chunk_size)):
    validate_x[ (i*chunk_size) : (i*chunk_size + chunk_size)] = extract_features(model_xception, validate_set_chunk)    

print('shape of validate_x: ', np.shape(validate_x))

# prepare train_y
train_y = train_set[:,1]
train_y = to_categorical(train_y)        

# prepare validate_y
validate_y = validate_set[:,1]
validate_y = to_categorical(validate_y) 

transfer_model = Sequential()

transfer_model.add(keras.Input(shape = (IMAGE_FEATURES_SIZE)))
transfer_model.add(Flatten())
transfer_model.add(Dense(100, activation = 'relu'))
transfer_model.add(Dense(16,  activation = 'relu'))
transfer_model.add(Dense(2,   activation = 'softmax')) # The last layer has 2 units beacuse we have 2 classes (Dog and cat)

transfer_model.summary()

import tensorflow as tf

# compile and fit the model
opt = tf.keras.optimizers.RMSprop(learning_rate=0.0002, rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    name="RMSprop")

transfer_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

EPOCHS = 100

history = transfer_model.fit(x = train_x , y= train_y, batch_size = 16, epochs = EPOCHS, 
                             callbacks = [early_stop, learning_rate_reduction],
                             validation_data = (validate_x, validate_y))



#Visualize the training process
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, EPOCHS, 1))
ax1.set_yticks(np.arange(0, 0.14, 0.02))
ax1.legend()

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, EPOCHS, 1))
ax2.legend()

legend = plt.legend(loc='best')
plt.tight_layout()
plt.show()

#Predict
test_x = np.zeros((len(test_set), 1000)) 
test_y = np.zeros((len(test_set) ))

# extract features from test_set and save it into test_x
for i, test_set_chunk in enumerate (chunks(test_set, chunk_size)):
    test_x[ (i*chunk_size) : (i*chunk_size + chunk_size)] = extract_features(model_xception, test_set_chunk)    

# prepare test_y
test_y = test_set[:,1]
test_y = to_categorical(test_y) 

assert(np.shape(test_y)  == (np.shape(test_set)[0], 2))

# predict and estimate the accuracy
_, accuracy = transfer_model.evaluate(test_x, test_y)   
print('accuracy on test set = ',  round((accuracy * 100),2 ), '% ') 

#Predict Class of Random Image
sample = random.choice(test_set)[0]
randomimage = image.load_img(imgs_path + sample)
plt.imshow(randomimage)

loaded_image = _load_image(imgs_path + sample)
extracted_feat = model_xception.predict(loaded_image) 
pred = transfer_model.predict(extracted_feat)

pred = pred[0]  # convert to array
if pred[0] >= pred[1]:
    prediction_class = 'Okpunta'
    prediction_percentage = pred[0]
else:
    prediction_class = 'Puntanok'
    prediction_percentage = pred[1]

print('I am ',int(prediction_percentage*100), '% sure that I am a ', prediction_class, '!', sep = '')


