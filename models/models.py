'''
Models for estimating z.
'''
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, LeakyReLU, Lambda, Dropout
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Dense, Concatenate
from keras.initializers import Constant
import numpy as np

def newby(shape, activation='softplus'):
    '''
    Inspired from Newby et al. (PNAS, 2018).
    '''
    # Input layer
    inputs = Input(shape=shape)

    # Parallel convolutional paths
    conv1 = Conv2D(3, (3, 3), activation=activation, padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(3, (5, 5), activation=activation, padding='same', strides=2)(inputs)
    conv3 = Conv2D(3, (9, 9), activation=activation, padding='same', strides=2)(inputs)
    concat1 = Concatenate()([pool1, conv2, conv3])

    # Second layer: two parallel convolutional layers
    conv2_1 = Conv2D(6, (7, 7), activation=activation, padding='same', dilation_rate=2)(concat1)
    conv2_2 = Conv2D(6, (3, 3), activation=activation, padding='same', dilation_rate=2)(concat1)
    concat2 = Concatenate()([conv2_1, conv2_2])

    # Third layer: single convolutional layer
    conv3 = Conv2D(2, (5, 5), activation=activation, padding='same')(concat2)

    # Final regression layers
    # Global average pooling
    gap = GlobalAveragePooling2D()(conv3)
    dense = Dense(128, activation=activation)(gap)
    output = Dense(1, activation='linear')(dense)

    # Create the model
    model = Model(inputs=inputs, outputs=output)
    return model

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

def simple_conv_2dense(shape):
    model = Sequential([  # tuned model, but I'm not sure, final receptive fields are too small
        Conv2D(32, (3, 3), activation='leaky_relu', input_shape=shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='leaky_relu'),
        #MaxPooling2D((2, 2)),
        GlobalAveragePooling2D(),
        Flatten(),
        Dense(128, activation='leaky_relu'),
        Dense(32, activation='leaky_relu'),
        Dense(1)
    ])
    return model

def simple_conv_rectified(shape):
    model = Sequential([  # tuned model, but I'm not sure, final receptive fields are too small
        Conv2D(32, (3, 3), activation='leaky_relu', input_shape=shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='leaky_relu'),
        #MaxPooling2D((2, 2)),
        GlobalAveragePooling2D(),
        Flatten(),
        Dense(128, activation='leaky_relu'),
        Dense(1, activation='softplus', use_bias=False),
        Dense(1, kernel_initializer=Constant(-1.), trainable=False) # works for focus on top only
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

def conv(shape):
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=shape),
        LeakyReLU(),  # Activation après BN
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3)),
        LeakyReLU(),  # Activation après BN
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1)
    ])
    return model

def two_point_model(shared_model):
    '''
    A model applied on two consecutive images
    '''
    _, n, m, _ = shared_model.input_shape

    # Define the input
    input_shape = (n, m, 2)
    inputs = tf.keras.Input(shape=input_shape)

    # Split the input into two channels
    channel_1 = layers.Lambda(lambda x: x[..., 0:1], name='first_image')(inputs)
    channel_2 = layers.Lambda(lambda x: x[..., 1:2], name='second_image')(inputs)

    # Apply the shared Sequential model to both channels
    output_1 = shared_model(channel_1)
    output_2 = shared_model(channel_2)

    # Combine the outputs
    outputs = layers.Concatenate()([output_1, output_2])

    # Create the final model
    return Model(inputs, outputs)

def extract_single_model(model):
    '''
    Extracts the shared model underlying a duplicate model.
    '''
    first_channel_input = model.input  # This is the original input layer
    first_image = model.get_layer('first_image')(first_channel_input)  # Extract the first channel

    # Recreate the submodel that only processes the first channel
    # Reuse the shared model without retraining
    # For this, you'll reference the layers from the original model
    # Here, we're assuming that 'model' is the shared submodel for each channel.

    # Using the same shared model for processing the first channel
    shared_model = model.get_layer('model')  # This is the shared submodel

    # Apply the shared model to the first channel
    first_channel_output = shared_model(first_image)

    # Create the new submodel for the first channel
    return Model(inputs=first_image, outputs=first_channel_output)

models = {'newby': newby,
            'efficient_net': efficient_net,
            "dense_model": dense_model,
            'conv': conv,
            'convmax': convmax,
            'batch_conv': batch_conv,
            'simple_conv': simple_conv,
            'simple_conv_2dense': simple_conv_2dense,
            "simple_conv_rectified": simple_conv_rectified}

if __name__ == '__main__':
    shared_model = efficient_net((96, 96, 1))
    model = two_point_model(shared_model)
    model.summary()
    extract_single_model(model).summary()
