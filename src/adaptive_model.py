import tensorflow as tf
import cv2
import random 
import numpy as np
import itertools
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Activation, Flatten, Dropout, Conv2DTranspose, LeakyReLU, Concatenate, Lambda

from tensorflow.keras import backend as K


def define_model(input_layer, output_layer):
    model = Model(inputs=[input_layer], outputs=[output_layer])
    return model


def conv_layer(input_tensor, n_filters, kernel_size=3, batchnorm=True, name=None):
    if name:
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
               padding='same', trainable=True, name='first_layer')(input_tensor)
    else:
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


def flatten_layer(layer):
    output = Flatten()(layer)
    return output


def transfer_layer(layer):
    output = GlobalAveragePooling2D()(layer)
    return output


def output_layer(layer):
    output = Dense(2, activation='softmax')(layer)
    return output


def unet_layer(layer, kernel_size=3, n_filters=16):
    c1 = conv_layer(layer, n_filters, kernel_size=kernel_size, batchnorm=True, name='first_layer')
    p1 = MaxPooling2D((2, 2))(c1)
    output = Dropout(0.1)(p1)
    return output


def encoder_layer(layer, kernel_size=3, n_filters=32):
    c1 = Conv2D(n_filters, (kernel_size, kernel_size), activation='relu', padding='same', name='first_layer')(layer)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(n_filters, (kernel_size, kernel_size), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(n_filters, (kernel_size, kernel_size), activation='relu', padding='same')(p1)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(n_filters, (kernel_size, kernel_size), activation='relu', padding='same', name='second_layer')(c2)
    output = Concatenate(axis=-1)([UpSampling2D((2, 2))(c2), c1])
    return output


def gabor_filter(model):
    gabor_filters = []
    weightMatrix = [(np.empty(model.layers[1].get_weights()[0].shape, dtype=float)), 
                    (np.empty(model.layers[1].get_weights()[1].shape, dtype=float))]
    orient = 1
    start_lam = 1
    stop_lam = 1024
    num_lam = 64
    lambdMatrix = np.linspace(start_lam, stop_lam, num_lam)
    lambdMatrix = np.resize(lambdMatrix, [weightMatrix[0].shape[3]])
    orientMatrix = np.array([(j/orient)*np.pi for j in range(orient)])
    orientMatrix = np.resize(orientMatrix, [weightMatrix[0].shape[3]])
    lamOrientMatrix = list(itertools.product(orientMatrix, lambdMatrix))
    
    theta = np.arange(0, np.pi, np.pi / weightMatrix[0].shape[3])
    lambd = np.arange(0, np.pi, np.pi / weightMatrix[0].shape[3])
    
    ksize = weightMatrix[0].shape[0]
    for i in range(weightMatrix[0].shape[3]):
        gabor_filter  = cv2.getGaborKernel((ksize, ksize), 5.0, theta[random.randrange(1, len(theta) - 1)], lambd[random.randrange(1, len(lambd) - 1)], 1, 0, ktype=cv2.CV_32F)
        weightMatrix[0][:,:,0,i] = gabor_filter        
        gabor_filters.append(gabor_filter)
    # Set Bias
    weightMatrix[1][:] = 1
    return weightMatrix, gabor_filters
            
    
def gabor_layer(layer, kernel_size=7, n_filters=32):
    conv = Conv2D(n_filters, (kernel_size, kernel_size), padding='same', activation='relu', trainable=False, name='first_layer')(layer)
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
    return outputs
