from keras.layers import Input
from keras.models import Model

from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping

from models import getUnetOuput
from utils import split_train_test, splitData

import numpy as np
import argparse
import os

img_width=img_height=128

input_img = Input((img_height, img_width, 1))

parser = argparse.ArgumentParser()
parser.add_argument('--img-root', required=True)
parser.add_argument('--mask-root', required = True)

args = parser.parse_args()

#call splitData
if not os.path.exists('train_files.txt'):
    splitData(args.img_root, args.mask_root)

X_train, X_test, y_train, y_test = split_train_test('train_files.txt', 'test_files.txt')

outputs = getUnetOuput(input_img)

model = Model(inputs=[input_img], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=32, epochs=30,
            callbacks=[EarlyStopping(patience=10, verbose=1),
            ModelCheckpoint('saved-unet-model.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)])
