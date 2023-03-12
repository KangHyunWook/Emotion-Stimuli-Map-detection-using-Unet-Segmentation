from keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, Conv2DTranspose
from keras.models import Model

from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

from models import getUnetOuput
from utils import preprocessData

import numpy as np
import argparse
import os

img_width=img_height=128

input_img = Input((img_height, img_width, 1))

parser = argparse.ArgumentParser()
parser.add_argument('--img-root', required=True)
parser.add_argument('--mask-root', required = True)

args = parser.parse_args()

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

outputs = getUnetOuput(input_img)

model = Model(inputs=[input_img], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=30,
            callbacks=[ModelCheckpoint('saved-unet-model.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)])








#
