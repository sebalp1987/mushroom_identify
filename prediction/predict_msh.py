from resources import STRING
import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plot
from keras.models import load_model

test = ['../data_input/test/{}'.format(i) for i in os.listdir(STRING.test)]
model = load_model(STRING.model_path + 'model.h5')

X = [] # images
nrows = 150
ncolumns = 150

for image in test:
    X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))

X = np.array(X)
datagen = ImageDataGenerator(rescale=1./255)

columns = 5
i = 0
text_label = []
pred_score = []
plot.figure(figsize=(30, 20))
for batch in datagen.flow(X, batch_size=1):
    pred = model.predict(batch)
    if pred > 0.5:
        text_label.append('Destroying Angel')
        pred_score.append(pred)
    else:
        text_label.append('Kantarell')
        pred_score.append(1-pred)
    plot.subplot(5/columns + 1, columns, i + 1)
    plot.title('This is a ' + text_label[i] + ' with probability ' + str(pred_score[i]), fontsize=8)
    imgplot = plot.imshow(batch[0])
    i += 1
    if i % 10 == 0:
        break

plot.show()
