from resources import STRING
import os
import gc
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras import layers
from keras import optimizers
from keras import models
from keras.utils import to_categorical

train_imgs = ['../data_input/train/{}'.format(i) for i in os.listdir(STRING.train)]

# Resize
nrows = 150
ncolumns = 150
channels = 3

X = []  # images
y = []  # labels

mushrooms = STRING.mushrooms

for image in train_imgs:
    X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))
    value = -1
    for key, val in mushrooms.items():
        if key in image:
            value = val
    y.append(value)

X = np.array(X)
y = np.array(y)


x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
print(y_train)
print(y_val)
del X, y
gc.collect()

ntrain = len(x_train)
nval = len(x_val)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(nrows, ncolumns, channels)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(len(mushrooms), activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

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
