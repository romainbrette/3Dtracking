import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 28*28) / 255.0
X_test = X_test.reshape(-1, 28*28) / 255.0

# Build a simple model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(28*28,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Build a model with batch normalization
model = keras.Sequential([
    # First layer
    layers.Dense(128, input_shape=(28 * 28,)),
    layers.BatchNormalization(),  # Add batch normalization
    layers.Activation('relu'),  # Activation after normalization

    # Second layer
    layers.Dense(128),
    layers.BatchNormalization(),  # Add batch normalization
    layers.Activation('relu'),  # Activation after normalization

    # Output layer
    layers.Dense(10),
    layers.BatchNormalization(),  # Add batch normalization
    layers.Activation('softmax')  # Activation after normalization
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)
