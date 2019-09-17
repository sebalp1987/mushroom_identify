from resources import STRING
import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plot
import matplotlib.image as mpimg
from keras.models import load_model


test = ['../data_input/test/{}'.format(i) for i in os.listdir(STRING.test)]
print(test)
model = load_model(STRING.model_path + 'model.h5')
mushrooms = STRING.mushrooms
info_mush = STRING.explain_type
X = [] # images
nrows = 150
ncolumns = 150

for image in test:
    X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))

X = np.array(X)
datagen = ImageDataGenerator(rescale=1./255)

text_label = []
pred_score = []


j = 0
for i, batch in enumerate(datagen.flow(X, batch_size=1)):
    pred = model.predict(batch)
    pred = [item for sublist in pred for item in sublist]
    print(pred)
    pred_score = max(pred)
    ind_max = np.argmax(pred)
    print(ind_max)
    type_m = ""
    info = ""
    for key, val in mushrooms.items():
        if val == ind_max:
            type_m = key
            info = info_mush.get(val)
            break
    print(info)

    plot.title('This is a ' + type_m + ' with probability ' + str(pred_score), fontsize=8)
    img = mpimg.imread(test[i])
    plot.imshow(img)
    plot.show()
    j += 1
    if j == len(test):
        break
