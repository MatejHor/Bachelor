from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, MaxPool2D, Flatten, LeakyReLU


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


def cnn2(shape, num_classes):
    input_layer = Input(shape=shape)

    # First layer
    conv1 = Conv2D(64, (3, 3), padding='same', activation='linear')(input_layer)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = MaxPool2D((2, 2))(conv1)

    conv1 = Dropout(0.25)(conv1)
    
    # Second layer
    conv2 = Conv2D(128, (3, 3), padding='same', activation='linear')(conv1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = MaxPool2D((2, 2))(conv2)
    
    conv2 = Dropout(0.25)(conv2)

    # Third layer
    conv3 = Conv2D(256, (3, 3), padding='same', activation='linear')(conv2)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = MaxPool2D((2, 2))(conv3)
    
    conv3 = Dropout(0.25)(conv3)

    # Fourth layer
    conv4 = Conv2D(512, (3, 3), padding='same', activation='linear')(conv3)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    conv4 = MaxPool2D((2, 2))(conv4)
    
    conv4 = Dropout(0.25)(conv4)

    # Fifth layer
    conv5 = Conv2D(512, (3, 3), padding='same', activation='linear')(conv4)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = MaxPool2D((2, 2))(conv5)
    
    conv5 = Dropout(0.25)(conv5)

    dense1 = Flatten()(conv5)
    dense1 = Dense(4096, activation='linear')(dense1)
    dense1 = LeakyReLU(alpha=0.1)(dense1)

    dense2 = Dense(4096, activation='linear')(dense1)
    dense2 = LeakyReLU(alpha=0.1)(dense2)

    output = Dense(num_classes, activation='softmax')(dense2)

    model = Model(input=[input_layer], output=[output])
    return model
