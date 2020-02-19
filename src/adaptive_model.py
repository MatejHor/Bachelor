import tensorflow as tf
import cv2
import numpy as np
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.layers import Activation, Flatten, Dropout, Conv2DTranspose, LeakyReLU, Concatenate, Lambda

from tensorflow.keras import backend as K


def define_model(input_layer, output_layer):
    model = Model(inputs=[input_layer], outputs=[output_layer])
    return model


def conv_layer(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
               padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
               padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def input_layer(shape):
    return Input(shape=shape, name='img')


def output_layer(layer):
    output = Flatten()(layer)
    output = Dense(2, activation='softmax')(output)
    return output


def unet_layer(layer):
    c1 = conv_layer(layer, 16, kernel_size=3, batchnorm=True)
    p1 = MaxPooling2D((2, 2))(c1)
    output = Dropout(0.1)(p1)
    return output


def encoder_concatenate_layer(layer):
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(layer)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    output = Concatenate(axis=-1)([UpSampling2D((2, 2))(c2), c1])
    return output


def encoder_multiple_upsampling_layer(layer):
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    output = UpSampling2D((2, 2))(x)
    return output


def gabor_filter(shape, dtype=None):
    filters = []
    ksize = shape[0]
    for theta in np.arange(0, np.pi, np.pi / shape[3]):
        params = {'ksize':(ksize, ksize), 'sigma':1.0, 'theta':theta, 'lambd':15.0,
                  'gamma':0.02, 'psi':0, 'ktype':cv2.CV_32F}
        kernel = cv2.getGaborKernel(**params)
        kernel /= 1.5*kernel.sum()
        
        reshaped_kernel = []
        for row in kernel:
            row_kernel = []
            for column in row:
                row_kernel.append([[column],[column],[column]])
            reshaped_kernel.append(row_kernel)
    return K.variable(np.array(reshaped_kernel), dtype='float32')
#     f = np.array([
#             [[[1],[1],[1]], [[0],[0],[0]], [[-1],[-1],[-1]]],
#             [[[1],[1],[1]], [[0],[0],[0]], [[-1],[-1],[-1]]],
#             [[[1],[1],[1]], [[0],[0],[0]], [[-1],[-1],[-1]]]
#         ])


def gabor_layer(layer):
    layer = Conv2D(filters=1, kernel_size = 3, kernel_initializer=gabor_filter, strides=2, padding='valid')(layer)
    return layer


def unet_model(layer, n_filters=16, dropout=0.1, batchnorm=True):
    
    c1 = conv_layer(layer, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv_layer(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv_layer(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv_layer(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv_layer(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv_layer(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv_layer(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv_layer(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv_layer(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    output = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    return output_layer(output)
