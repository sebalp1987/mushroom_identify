from resources import STRING
import os
import gc
import cv2
import numpy as np

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

print(X.shape)