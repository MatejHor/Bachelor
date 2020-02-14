import tensorflow as tf
import cv2, numpy as np

from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, MaxPool2D, Flatten, LeakyReLU
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K


def cnn1(shape, num_classes):
    input_layer = Input(shape=shape)

    # First layer
    conv1 = Conv2D(16, (3, 3), data_format="channels_first", padding='same', activation='linear')(input_layer)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)

    conv1 = Dropout(0.25)(conv1)
    
    conv1 = Conv2D(16, (3, 3), data_format="channels_first", padding='same', activation='linear')(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    
    conv1 = Dropout(0.25)(conv1)

    conv1 = Conv2D(16, (2, 2), data_format="channels_first", padding='same', activation='linear')(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    
    conv1 = Dropout(0.25)(conv1)

    # Second layer
    conv2 = Conv2D(32, (3, 3), data_format="channels_first", padding='same', activation='linear')(conv1)
    conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)

    conv2 = Dropout(0.25)(conv2)
    
    conv2 = Conv2D(32, (3, 3), data_format="channels_first", padding='same', activation='linear')(conv2)
    conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    
    conv2 = Dropout(0.25)(conv2)

    conv2 = Conv2D(32, (2, 2), data_format="channels_first", padding='same', activation='linear')(conv2)
    conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    
    conv2 = Dropout(0.25)(conv2)

    # Third layer
    conv3 = Conv2D(64, (3, 3), data_format="channels_first", padding='same', activation='linear')(conv2)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    
    conv3 = Dropout(0.25)(conv3)

    conv3 = Conv2D(64, (3, 3), data_format="channels_first", padding='same', activation='linear')(conv3)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    
    conv3 = Dropout(0.25)(conv3)

    conv3 = Conv2D(64, (2, 2), data_format="channels_first", padding='same', activation='linear')(conv3)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    
    conv3 = Dropout(0.25)(conv3)

    # Fourth layer
    conv4 = Conv2D(128, (3, 3), data_format="channels_first", padding='same', activation='linear')(conv3)
    conv4 = BatchNormalization(axis=1)(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)

    conv4 = Dropout(0.25)(conv4)
    
    conv4 = Conv2D(128, (3, 3), data_format="channels_first", padding='same', activation='linear')(conv4)
    conv4 = BatchNormalization(axis=1)(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    
    conv4 = Dropout(0.25)(conv4)

    conv4 = Conv2D(128, (3, 3), data_format="channels_first", padding='same', activation='linear')(conv4)
    conv4 = BatchNormalization(axis=1)(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    
    conv4 = Dropout(0.25)(conv4)

    # Fifth layer
    conv5 = Conv2D(256, (3, 3), data_format="channels_first", padding='same', activation='linear')(conv4)
    conv5 = BatchNormalization(axis=1)(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)

    conv5 = Dropout(0.25)(conv5)
    
    conv5 = Conv2D(256, (3, 3), data_format="channels_first", padding='same', activation='linear')(conv5)
    conv5 = BatchNormalization(axis=1)(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    
    conv5 = Dropout(0.25)(conv5)

    conv5 = Conv2D(256, (3, 3), data_format="channels_first", padding='same', activation='linear')(conv5)
    conv5 = BatchNormalization(axis=1)(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    
    conv5 = Dropout(0.25)(conv5)

    # Sixth layer
    conv6 = Conv2D(128, (1, 1), data_format="channels_first", padding='same', activation='linear')(conv5)

    dense1 = Flatten()(conv6)
    output = Dense(num_classes, activation='softmax')(dense1)

    model = Model(input=[input_layer], output=[output])
    return model


def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x


def get_unet(shape, n_filters = 16, dropout = 0.1, batchnorm = True):
    """Function to define the UNET Model"""
    input_img = Input(shape, name='img')    
   
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    outputs = Flatten()(outputs)
    outputs = Dense(2, activation='softmax')(outputs)
    
    model = Model(inputs=[input_img], outputs=[outputs])
    return model



def vgg_16(shape, num_classes):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=shape))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


def autoencoder(shape, num_classes = 2):
    input_img = Input(shape=shape)  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    decoded = Flatten()(decoded)
    autoencoder = Dense(num_classes, activation="softmax")(decoded)
    
    model = Model(inputs=[input_img], outputs=[autoencoder])
    return model