# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 15:51:06 2021

@author: ilhan
"""

import os
import zipfile
import numpy as np 
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import random
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, applications, optimizers, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation,GlobalMaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model

#np.random.seed(9)
#tf.random.set_seed(9)

#for dirname, _, filenames in os.walk('../makina_ogrenmesi/inceptionv3/'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

image_size = 128

input_shape = (image_size, image_size, 3)

epochs = 100


batch_size = 50
        
weights_file = '../inceptionv3_punta/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (image_size, image_size, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(weights_file)
#pre_trained_model.summary()

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
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


#print(f"We have total {len(os.listdir('../makina_ogrenmesi/inceptionv3/train/'))} images in our training data.")

#filenames = os.listdir('../inceptionv3/train')
#labels = [str(fname)[:3] for fname in filenames]
#train_df = pd.DataFrame({'filename': filenames, 'label': labels})
#train_df.head()

#print((train_df['label']).value_counts())

#train_set_df, dev_set_df = train_test_split(train_df[['filename', 'label']], test_size=0.3, random_state = 42, shuffle=True, stratify=train_df['label'])
#print(train_set_df.shape, dev_set_df.shape)


#print('Training Set image counts:')
#print(train_set_df['label'].value_counts())
#print('Validation Set image counts:')
#print(dev_set_df['label'].value_counts())

filenames = os.listdir("../VGG16_punta/working/train")
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

train_df, validate_df = train_test_split(df, test_size=0.1)
train_df = train_df.reset_index()
validate_df = validate_df.reset_index()

# validate_df = validate_df.sample(n=100).reset_index() # use for fast testing code purpose
# train_df = train_df.sample(n=1800).reset_index() # use for fast testing code purpose

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen  = ImageDataGenerator( rescale = 1.0/255 )

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    directory="../VGG16_punta/working/train/", 
    x_col='filename',
    y_col='category',
    target_size=(image_size,image_size),
    class_mode='categorical',
    )

validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    directory="../VGG16_punta/working/train/", 
    x_col='filename',
    y_col='category',
    target_size=(image_size,image_size),
    class_mode='categorical',   
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
x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense  (1, activation='sigmoid')(x)           

model = Model( pre_trained_model.input, x ) 

model.summary()
'''



import tensorflow as tf
opt = tf.keras.optimizers.RMSprop(learning_rate=0.0002, rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    name="RMSprop")

model.compile(optimizer = opt, 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])

model.summary()


#history = model.fit(
#            train_generator,
#            validation_data = validation_generator,
#            steps_per_epoch = 20,
#            epochs = epochs,
#            validation_steps = 20)

# fine-tune the model
checkpoint_path = 'model_0_vgg16.h5'

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


test_filenames = os.listdir("../VGG19_gazalti/working/test/")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "../VGG19_gazalti/working/test/", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    batch_size=32,
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

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs   = range(len(acc))

plt.plot(epochs, acc, label="Training accuracy")
plt.plot(epochs, val_acc, label="Validation accuracy")
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, label="Training loss")
plt.plot(epochs, val_loss, label="Validation loss")
plt.title('Training and validation loss')
plt.legend()
plt.show()

loss, accuracy = model.evaluate_generator(validation_generator)
print("Test: accuracy = %f  ;  loss = %f " % (accuracy, loss))

#dev_true = dev_set_df['label'].map({'Yirtik': 1, "Okparca": 0})
#dev_predictions =  model.predict_generator(validation_generator)
#dev_set_df['pred'] = np.where(dev_predictions>0.5, 1, 0)
#dev_pred = dev_set_df['pred']
#dev_set_df.head()


