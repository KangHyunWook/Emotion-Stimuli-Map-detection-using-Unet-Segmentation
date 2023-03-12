from keras import preprocessing
from utils import getImagePaths
from sklearn.model_selection import train_test_split
from skimage import transform

import numpy as np
import random
import os
import matplotlib.pyplot as plt

if not os.path.exists('X_train.npy'):

    X_train, X_valid, X_test, y_train, y_valid, y_test = preprocessData(args)

    np.save('X_train.npy', X_train)
    np.save('X_valid.npy', X_valid)
    np.save('X_test.npy', X_test)

    np.save('y_train.npy', y_train)
    np.save('y_valid.npy', y_valid)
    np.save('y_test.npy', y_test)

X_train=np.load('X_train.npy')
X_valid=np.load('X_valid.npy')
X_test=np.load('X_test.npy')
y_train=np.load('y_train.npy')
y_valid=np.load('y_valid.npy')
y_test=np.load('y_test.npy')

ix = random.randint(0, len(X_test))
has_mask = y_test[ix].max() > 0

img_width=img_height=128

from keras.models import Model

from keras.layers import Input

from models import getUnetOuput

input_img = Input((img_height, img_width, 1))

outputs = getUnetOuput(input_img)

model = Model(inputs=[input_img], outputs=[outputs])

model.load_weights('saved-unet-model.h5')


preds = model.predict(X_test, verbose=1)

preds_t = (preds > 0.5).astype(np.uint8)

fig, ax = plt.subplots(1, 4, figsize=(20,10))
from keras import preprocessing
from skimage import transform

test_img = preprocessing.image.load_img('/home/jeff/data/EmotionROI/images/joy/317.jpg', grayscale='True')
test_img = preprocessing.image.img_to_array(test_img)
test_img = transform.resize(test_img, (128, 128,1), mode='constant', preserve_range = True)
test_img/=255.

ax[0].imshow(test_img)

test_img_ground_truth = preprocessing.image.load_img('/home/jeff/data/EmotionROI/ground_truth/joy/317.jpg', grayscale='True')
test_img_ground_truth = preprocessing.image.img_to_array(test_img_ground_truth)
test_img_ground_truth = transform.resize(test_img_ground_truth, (128, 128,1), mode='constant', preserve_range=True)
test_img_ground_truth /= 255.

ax[1].imshow(test_img_ground_truth)
test_img=np.expand_dims(test_img, axis=0)
print(test_img.shape)

# exit()
preds = model.predict([test_img], verbose=1)
print(preds.shape)
# exit()
ax[2].imshow(preds[0], vmin=0, vmax=1)
# ax[3].imshow(preds_t[ix].squeeze(), vmin=0, vmax=1)

plt.show()
# print('ix:', ix)
















#
