
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

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

def getUnetOuput(x):
    c1 = conv2d_block(16, x)
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

#
