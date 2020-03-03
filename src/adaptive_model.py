import tensorflow as tf
import cv2
import numpy as np
import itertools
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
               padding='same', trainable=True)(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
               padding='same', trainable=True)(input_tensor)
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


def unet_layer(layer, kernel_size=3, n_filters=16):
    c1 = conv_layer(layer, n_filters, kernel_size=kernel_size, batchnorm=True)
    p1 = MaxPooling2D((2, 2))(c1)
    output = Dropout(0.1)(p1)
    return output


def encoder_layer(layer, kernel_size=3, n_filters=32):
    c1 = Conv2D(n_filters, (kernel_size, kernel_size), activation='relu', padding='same')(layer)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(n_filters, (kernel_size, kernel_size), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(n_filters, (kernel_size, kernel_size), activation='relu', padding='same')(p1)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(n_filters, (kernel_size, kernel_size), activation='relu', padding='same')(c2)
    output = Concatenate(axis=-1)([UpSampling2D((2, 2))(c2), c1])
    return output


# def encoder_multiple_upsampling_layer(layer):
#     x1 = Conv2D(16, (3, 3), activation='relu', padding='same')(layer)
#     x2 = MaxPooling2D((2, 2), padding='same')(x1)
#     x3 = Conv2D(8, (3, 3), activation='relu', padding='same')(x2)
#     x4 = MaxPooling2D((2, 2), padding='same')(x3)
#     x5 = Conv2D(8, (3, 3), activation='relu', padding='same')(x4)
#     encoded = MaxPooling2D((2, 2), padding='same')(x5)

#     x6 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
#     x7 = UpSampling2D((2, 2))(x6)
#     x8 = Conv2D(8, (3, 3), activation='relu', padding='same')(x7)
#     x9 = UpSampling2D((2, 2))(x8)
#     x10 = Conv2D(16, (3, 3), activation='relu')(x9)
#     output = UpSampling2D((2, 2))(x10)
#     return output


def gabor_filter(model):
    weightMatrix = [(np.empty(model.layers[1].get_weights()[0].shape, dtype=float)), 
                    (np.empty(model.layers[1].get_weights()[1].shape, dtype=float))]
    orient = 6
    start_lam = 1
    stop_lam = 1
    num_lam = 1
    lambdMatrix = np.linspace(start_lam, stop_lam, num_lam)
    lambdMatrix = np.resize(lambdMatrix, [weightMatrix[0].shape[3]])
    orientMatrix = np.array([(j/orient)*np.pi for j in range(orient)])
    orientMatrix = np.resize(orientMatrix, [weightMatrix[0].shape[3]])
    lamOrientMatrix = list(itertools.product(orientMatrix, lambdMatrix))
    ksize = weightMatrix[0].shape[0]
    for i in range(weightMatrix[0].shape[3]):
        weightMatrix[0][:,:,0,i] = cv2.getGaborKernel((ksize, ksize), 5.0, lamOrientMatrix[i][0],\
                                                      lamOrientMatrix[i][1], 1, 0, ktype=cv2.CV_32F)
    # Set Bias
    weightMatrix[1][:] = 1
    return weightMatrix
            
    
def gabor_layer(layer, kernel_size=7, n_filters=32):
    conv = Conv2D(n_filters, (kernel_size, kernel_size), padding='same', activation='relu', trainable=False)(layer)
    return conv


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
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    return output_layer(outputs)
