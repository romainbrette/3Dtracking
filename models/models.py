'''
Models for estimating z.
'''
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, LeakyReLU, Lambda, Dropout
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input

def efficient_net(shape):
    '''
    EfficientNet models expect their inputs to be float tensors of pixels with values in the [0-255] range.
    '''
    base_model = EfficientNetB0(include_top=False, weights='imagenet',
                                input_shape=(shape[0], shape[1], 3))  # Keep EfficientNet's input shape

    # Create the model
    inputs = Input(shape=shape)
    x = tf.image.grayscale_to_rgb(inputs)  # Expand grayscale to 3 channels in the input pipeline
    x = base_model(x, training=False)  # Use EfficientNet as base
    x = GlobalAveragePooling2D()(x)  # Add global average pooling
    x = Dense(128, activation='relu')(x)  # Optional dense layer
    outputs = Dense(1, activation='linear')(x)  # Regression output layer
    model = Model(inputs, outputs)
    return model

def dense_model(shape):
    model = Sequential([
        Flatten(input_shape=shape),
        Dense(64, activation='leaky_relu'),
        BatchNormalization(),
        Dense(64, activation='leaky_relu'),
        BatchNormalization(),
        Dense(1)
    ])
    return model

def convmax(shape):
    model = Sequential([ # tuned model, but I'm not sure, final receptive fields are too small
        Conv2D(75, (5, 5), activation='leaky_relu', input_shape=shape),
        MaxPooling2D((2, 2)),
        Conv2D(87, (5, 5), activation='leaky_relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(161, activation='leaky_relu'),
        Dense(1)
    ])
    return model

def simple_conv(shape):
    model = Sequential([  # tuned model, but I'm not sure, final receptive fields are too small
        Conv2D(32, (3, 3), activation='leaky_relu', input_shape=shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='leaky_relu'),
        #MaxPooling2D((2, 2)),
        GlobalAveragePooling2D(),
        Flatten(),
        Dense(128, activation='leaky_relu'),
        #Lambda(lambda x: x * span), # this actually slows down the training
        Dense(1)
    ])
    return model

def batch_conv(shape):
    model = Sequential([
        Conv2D(32, (3, 3), kernel_initializer='he_normal', input_shape=shape),
        BatchNormalization(),  # Doesn't seem to work
        LeakyReLU(),  # Activation après BN
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), kernel_initializer='he_normal'),
        BatchNormalization(),
        LeakyReLU(),  # Activation après BN
        MaxPooling2D((2, 2)),

        #GlobalAveragePooling2D(),
        Flatten(),
        #Dropout(0.5),
        Dense(128, activation='relu', kernel_initializer='he_normal'),
        Dense(1)
    ])
    return model
