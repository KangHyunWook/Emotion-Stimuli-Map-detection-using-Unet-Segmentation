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

img_width=img_height=128

input_img = Input((img_height, img_width, 1))

x = Conv2D(filters = 16, kernel_size = (3,3), kernel_initializer='he_normal', padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(filters = 16, kernel_size = (3,3), kernel_initializer = 'he_normal', padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
c1 = x
x = MaxPooling2D((2,2))(x)
x = Dropout(0.05)(x)
print('x:', x) #64x64x16

x = Conv2D(filters= 32, kernel_size = (3,3), kernel_initializer = 'he_normal', padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(filters = 32, kernel_size = (3,3), kernel_initializer = 'he_normal', padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
c2 = x
x = MaxPooling2D((2,2))(x)
x = Dropout(0.05)(x)

x = Conv2D(filters = 64, kernel_size = (3,3), kernel_initializer = 'he_normal', padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(filters = 64, kernel_size = (3,3), kernel_initializer = 'he_normal', padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
c3 = x
print('c3:', c3) #32x32x64
x = MaxPooling2D((2,2))(x)
x = Dropout(0.05)(x)
print('x:', x)

x = Conv2D(filters = 128, kernel_size = (3,3), kernel_initializer = 'he_normal', padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(filters = 128, kernel_size = (3,3), kernel_initializer = 'he_normal', padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
c4 = x
x = MaxPooling2D((2,2))(x)
x = Dropout(0.05)(x)

x= Conv2D(filters = 256, kernel_size = (3,3), kernel_initializer = 'he_normal', padding='same')(x)
x= BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(filters = 256, kernel_size = (3,3), kernel_initializer = 'he_normal', padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
c5 = x
print('c5:', c5)

u6 = Conv2DTranspose(128, (3,3), strides = (2,2), padding='same')(c5)
u6 = concatenate([u6, c4])
u6 = Dropout(0.05)(u6)
print('u6:', u6)

x = Conv2D(filters = 128, kernel_size = (3,3), kernel_initializer='he_normal', padding='same')(u6)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(filters = 128, kernel_size = (3,3), kernel_initializer = 'he_normal', padding='same')(x)
x = BatchNormalization()(x)
c6 = Activation('relu')(x)

u7 = Conv2DTranspose(64, kernel_size = (3,3), strides = (2,2), padding='same')(c6)
u7 = concatenate([u7, c3])
u7 = Dropout(0.05)(u7)

c7 = Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(u7)
c7 = BatchNormalization()(c7)
c7 = Activation('relu')(c7)
c7 = Conv2D(filters = 64, kernel_size = (3,3), kernel_initializer='he_normal', padding='same')(c7)
c7 = BatchNormalization()(c7)
c7 = Activation('relu')(c7)

u8 = Conv2DTranspose(32, kernel_size=(3,3), strides = (2,2), kernel_initializer='he_normal', padding='same')(c7)
# print('u8:', u8)
u8 = concatenate([u8, c2])
u8 = Dropout(0.05)(u8)
print('u8:', u8)
c8 = Conv2D(filters = 32, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(u8)
c8 = BatchNormalization()(c8)
c8 = Activation('relu')(c8)
c8 = Conv2D(filters = 32, kernel_size = (3,3), kernel_initializer='he_normal', padding='same')(c8)
c8 = BatchNormalization()(c8)
c8 = Activation('relu')(c8)

u9 = Conv2DTranspose(16, kernel_size=(3,3), strides = (2,2), kernel_initializer='he_normal', padding='same')(c8)
u9 = concatenate([u9, c1])
u9 = Dropout(0.05)(u9)

print('u9:', u9)
c9 = Conv2D(filters = 16, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(u9)
c9 = BatchNormalization()(c9)
c9 = Activation('relu')(c9)
c9 = Conv2D(filters=16, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(c9)
c9 = BatchNormalization()(c9)
c9 = Activation('relu')(c9)

outputs = Conv2D(1, (1,1), activation='sigmoid')(c9)

print('outputs:', outputs)

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

root=r'/home/jeff/data/EmotionROI/images'
maskRoot=r'/home/jeff/data/EmotionROI/ground_truth'

imgPathList=getImagePaths(root)

maskPathList=getImagePaths(maskRoot)

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

model = Model(inputs=[input_img], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=30,
            callbacks=[ModelCheckpoint('saved-unet-model.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)])








#
