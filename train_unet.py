from keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras import preprocessing
from skimage import transform
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

import os
import numpy as np
import argparse

img_width=img_height=128

input_img = Input((img_height, img_width, 1))

def conv2d_block(n_filters, x):
    x = Conv2D(filters = n_filters, kernel_size = (3,3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = n_filters, kernel_size = (3,3), kernel_initializer = 'he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def upsample(x, copy, n_filters):

    x = Conv2DTranspose(n_filters, (3,3), strides = (2,2), padding='same')(x)
    x = concatenate([x, copy])
    x = Dropout(0.05)(x)

    return x



def getImagePaths(root):
    items= os.listdir(root)
    pathList=[]
    for item in items:
        full_path = os.path.join(root, item)
        if os.path.isfile(full_path):
            pathList.append(full_path)
        else:
            pathList.extend(getImagePaths(full_path))
    return pathList

parser = argparse.ArgumentParser()
parser.add_argument('--img-root', required=True)
parser.add_argument('--mask-root', required = True)

args = parser.parse_args()

imgPathList=getImagePaths(args.img_root)
maskPathList=getImagePaths(args.mask_root)

img_path=r"/home/jeff/data/EmotionROI/images/anger"

X = np.zeros((len(imgPathList), img_height, img_width, 1), dtype=np.float32)
y = np.zeros((len(imgPathList), img_height, img_width, 1), dtype=np.float32)

for i in range(len(imgPathList)):
    # Load images
    img = preprocessing.image.load_img(imgPathList[i], grayscale='True')

    x_img = preprocessing.image.img_to_array(img)

    x_img = transform.resize(x_img, (128, 128, 1), mode = 'constant', preserve_range = True)

    mask = preprocessing.image.img_to_array(preprocessing.image.load_img(maskPathList[i], grayscale=True))
    mask = transform.resize(mask, (128, 128, 1), mode = 'constant', preserve_range = True)

    X[i] = x_img/255.0
    y[i] = mask/255.0

# Split train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

def getUnetOuput(x):
    c1 = conv2d_block(16, input_img)
    x = MaxPooling2D((2,2))(c1)
    x = Dropout(0.05)(x)
    c2 = conv2d_block(32, x)
    x = MaxPooling2D((2,2))(c2)
    x = Dropout(0.05)(x)
    c3 = conv2d_block(64, x)
    x = MaxPooling2D((2,2))(c3)
    x = Dropout(0.05)(x)
    c4 = conv2d_block(128, x)
    x = MaxPooling2D((2,2))(c4)
    x = Dropout(0.05)(x)
    c5 = conv2d_block(256, x)

    u6 = upsample(c5, c4, 128)
    c6 = conv2d_block(128, u6)

    u7 = upsample(c6, c3, 64)
    c7 = conv2d_block(64, u7)

    u8 = upsample(c7, c2, 32)
    c8 = conv2d_block(32, u8)

    u9 = upsample(c8, c1, 16)
    c9 = conv2d_block(16, u9)

    outputs = Conv2D(1, (1,1), activation='sigmoid')(c9)

    return outputs


outputs = getUnetOuput(input_img)

model = Model(inputs=[input_img], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=30,
            callbacks=[ModelCheckpoint('saved-unet-model.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)])








#
