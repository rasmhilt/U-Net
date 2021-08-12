import tensorflow as tf
import keras
from keras.layers import *

def chooseUnet(chosen_unet,IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS):
    if chosen_unet == 'UNet':
        return UNet(IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS)
    elif chosen_unet == 'UNetPlus':
        return UNetPlus(IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS) 
    elif chosen_unet == 'UNetMinus':
        return UNetMinus(IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS)
    elif chosen_unet == 'UNetMinustwo':
        return UNetMinustwo(IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS)

# Normal U-Net
def UNet(IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS):
    # Input (None, IMG_WIDTH, IMG_HEIGHT, 1)
    inp = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS])

    # First convolution layers. (None, IMG_WIDTH, IMG_HEIGHT, 64)
    conv1 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inp)
    conv1 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

    # Maxpool layer 1. (None, IMG_WIDTH/2, IMG_HEIGHT/2, 128)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    
    # Maxpool layer 2. (None, IMG_WIDTH/4, IMG_HEIGHT/4, 256)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    
    # Maxpool layer 3. (None, IMG_WIDTH/8, IMG_HEIGHT/8, 512)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(512, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    
    # Maxpool layer 4. (None, IMG_WIDTH/16, IMG_HEIGHT/16, 1024)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = tf.keras.layers.Conv2D(1024, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = tf.keras.layers.Conv2D(1024, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Upsample and concat with conv4, check picture, then convolutions for next upsample. (None, IMG_WIDTH/8, IMG_HEIGHT8, 512)
    up6 = tf.keras.layers.Conv2D(512, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = tf.keras.layers.Conv2D(512, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = tf.keras.layers.Conv2D(512, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    # Upsample and concat with conv3, check picture, then convolutions for next upsample. (None, IMG_WIDTH/4, IMG_HEIGHT/4, 256)
    up7 = tf.keras.layers.Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = tf.keras.layers.Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = tf.keras.layers.Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    # Upsample and concat with conv2, check picture, then convolutions for next upsample. (None, IMG_WIDTH/2, IMG_HEIGHT/2, 128)
    up8 = tf.keras.layers.Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = tf.keras.layers.Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = tf.keras.layers.Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    # Upsample and concat with conv1, check picture. (None, IMG_WIDTH, IMG_HEIGHT, 64)
    up9 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)

    # Last convolution layers. (None, IMG_WIDTH, IMG_HEIGHT, 1)
    conv9 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = tf.keras.layers.Conv2D(2, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = tf.keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = tf.keras.Model(inputs = inp, outputs = conv9)

    return model

# U-Net with one extra layer. Very load heavy
def UNetPlus(IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS):
    # Input (None, IMG_WIDTH, IMG_HEIGHT, 1)
    inp = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS])

    # First convolution layers. (None, IMG_WIDTH, IMG_HEIGHT, 64)
    conv1 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inp)
    conv1 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

    # Maxpool layer 1. (None, IMG_WIDTH/2, IMG_HEIGHT/2, 128)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    
    # Maxpool layer 2. (None, IMG_WIDTH/4, IMG_HEIGHT/4, 256)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    
    # Maxpool layer 3. (None, IMG_WIDTH/8, IMG_HEIGHT/8, 512)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(512, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    
    # Maxpool layer 4. (None, IMG_WIDTH/16, IMG_HEIGHT/16, 1024)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = tf.keras.layers.Conv2D(1024, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = tf.keras.layers.Conv2D(1024, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Maxpool layer 5.
    pool5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop5)
    conv6 = tf.keras.layers.Conv2D(2048, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5)
    conv6 = tf.keras.layers.Conv2D(2048, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    drop6 = Dropout(0.5)(conv6)

    # Upsample & concat #1
    up7 = tf.keras.layers.Conv2D(1024, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop6))
    merge7 = concatenate([drop5,up7], axis = 3)
    conv7 = tf.keras.layers.Conv2D(1024, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = tf.keras.layers.Conv2D(1024, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    # Upsample & concat #2
    up8 = tf.keras.layers.Conv2D(512, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv4,up8], axis = 3)
    conv8 = tf.keras.layers.Conv2D(512, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = tf.keras.layers.Conv2D(512, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    # Upsample & concat #3
    up9 = tf.keras.layers.Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv3,up9], axis = 3)
    conv9 = tf.keras.layers.Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = tf.keras.layers.Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    # Upsample & concat #4
    up10 = tf.keras.layers.Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv9))
    merge10 = concatenate([conv2,up10], axis = 3)
    conv10 = tf.keras.layers.Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
    conv10 = tf.keras.layers.Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)

    # Upsample & concat #5
    up11 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv10))
    merge10 = concatenate([conv1,up11], axis = 3)

    # Last convolution layers. (None, IMG_WIDTH, IMG_HEIGHT, 1)
    conv11 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
    conv11 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
    conv11 = tf.keras.layers.Conv2D(2, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)

    # NO RELU for last one
    conv11 = tf.keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv11)
    model = tf.keras.Model(inputs = inp, outputs = conv11)

    return model

# U-Net with one layer less.
def UNetMinus(IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS):
    # Input (None, IMG_WIDTH, IMG_HEIGHT, 1)
    inp = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS])

    # First convolution layers. (None, IMG_WIDTH, IMG_HEIGHT, 64)
    conv1 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inp)
    conv1 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

    # Maxpool layer 1. (None, IMG_WIDTH/2, IMG_HEIGHT/2, 128)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    
    # Maxpool layer 2. (None, IMG_WIDTH/4, IMG_HEIGHT/4, 256)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    
    # Maxpool layer 3. (None, IMG_WIDTH/8, IMG_HEIGHT/8, 512)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop3)
    conv4 = tf.keras.layers.Conv2D(512, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)

    # Upsample & concat #1
    up5 = tf.keras.layers.Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop4))
    merge5 = concatenate([drop3,up5], axis = 3)
    conv5 = tf.keras.layers.Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv5 = tf.keras.layers.Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    # Upsample & concat #2
    up6 = tf.keras.layers.Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv2,up6], axis = 3)
    conv6 = tf.keras.layers.Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = tf.keras.layers.Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    # Upsample & concat #3
    up7 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv1,up7], axis = 3)

    # Last convolution layers. (None, IMG_WIDTH, IMG_HEIGHT, 1)
    conv7 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = tf.keras.layers.Conv2D(2, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    # NO RELU for last one
    conv7 = tf.keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv7)
    model = tf.keras.Model(inputs = inp, outputs = conv7)

    return model

# U-Net with two layers less.
def UNetMinustwo(IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS):
    # Input (None, IMG_WIDTH, IMG_HEIGHT, 1)
    inp = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS])

    # First convolution layers. (None, IMG_WIDTH, IMG_HEIGHT, 64)
    conv1 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inp)
    conv1 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

    # Maxpool layer 1. (None, IMG_WIDTH/2, IMG_HEIGHT/2, 128)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    
    # Maxpool layer 2. (None, IMG_WIDTH/4, IMG_HEIGHT/4, 256)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)

    # Upsample & concat #1
    up4 = tf.keras.layers.Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop3))
    merge4 = concatenate([conv2,up4], axis = 3)
    conv4 = tf.keras.layers.Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge4)
    conv4 = tf.keras.layers.Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)

    # Upsample & concat #2
    up5 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv4))
    merge5 = concatenate([conv1,up5], axis = 3)

    # Last convolution layers. (None, IMG_WIDTH, IMG_HEIGHT, 1)
    conv5 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv5 = tf.keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = tf.keras.layers.Conv2D(2, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    # NO RELU for last one
    conv5 = tf.keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv5)
    model = tf.keras.Model(inputs = inp, outputs = conv5)

    return model