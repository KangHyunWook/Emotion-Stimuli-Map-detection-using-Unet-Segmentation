from keras import preprocessing
from utils import getImagePaths, preprocess_img
from sklearn.model_selection import train_test_split
from skimage import transform

import numpy as np
import random
import os
import matplotlib.pyplot as plt


img_width=img_height=128

from keras.models import Model

from keras.layers import Input

from models import getUnetOuput

input_img = Input((img_height, img_width, 1))

outputs = getUnetOuput(input_img)

model = Model(inputs=[input_img], outputs=[outputs])

model.load_weights('saved-unet-model.h5')

img_path='/home/jeff/data/EmotionROI/images/sadness/229.jpg'
mask_path='/home/jeff/data/EmotionROI/ground_truth/sadness/229.jpg'

#todo
test_img1 = preprocess_img(img_path)
test_img1 = np.expand_dims(test_img1, axis=0)

preds = model.predict(test_img1, verbose=1)

test_img1=test_img1[0]
fig, ax = plt.subplots(1, 3, figsize=(20,10))
from keras import preprocessing
from skimage import transform

ax[0].imshow(test_img1)
ax[0].set_title('original image')

test_img_ground_truth = preprocess_img(mask_path)

ax[1].imshow(test_img_ground_truth)
ax[1].set_title('ground-truth image')

ax[2].imshow(preds[0], vmin=0, vmax=1)
ax[2].set_title('predicted ESM')

plt.show()
# print('ix:', ix)
















#
