from resources import STRING
import os
import gc
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import EarlyStopping
from keras import layers
from keras import optimizers
from keras import models


train_angel = ['../data_input/train/{}'.format(i) for i in os.listdir(STRING.train) if 'angel' in i]
train_kantarell = ['../data_input/train/{}'.format(i) for i in os.listdir(STRING.train) if 'kantarell' in i]

train_imgs = train_angel + train_kantarell
del train_angel, train_kantarell
gc.collect()

# Resize
nrows = 150
ncolumns = 150
channels = 3

X = [] # images
y = [] # labels

for image in train_imgs:
    X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))
    if 'angel' in image:
        y.append(1)
    elif 'kantarell' in image:
        y.append(0)

X = np.array(X)
y = np.array(y)

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
del X, y
gc.collect()

ntrain = len(x_train)
nval = len(x_val)

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(nrows, ncolumns, channels)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

# preprocess image
train_datagen = ImageDataGenerator(rescale=1. / 255,  # Scale image to (0, 1)
                                   rotation_range=40,  # the other params randomly apply transforms to generalize
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
val_generator = val_datagen.flow(x_val, y_val, batch_size=32)

history = model.fit(train_generator, steps_per_epoch=10, epochs=64, validation_data=val_generator,
                    validation_steps=5, callbacks=[EarlyStopping(patience=2)])


model.save(STRING.model_path + 'model.h5')