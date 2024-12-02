'''
Testing the batch normalizatino problem.
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Create a synthetic dataset
def create_dataset(num_samples, image_size):
    # Generate random grayscale images
    images = np.random.rand(num_samples, image_size, image_size, 1).astype(np.float32)
    # Generate targets as the sum of pixel values
    targets = np.sum(images, axis=(1, 2, 3)).astype(np.float32)
    return images, targets

# Hyperparameters
image_size = 32  # Size of the images (32x32)
num_samples = 100000  # Total number of samples
train_split = 0.8  # 80% training data

# Generate dataset
images, targets = create_dataset(num_samples, image_size)
split_idx = int(num_samples * train_split)

# Split into training and testing sets
x_train, y_train = images[:split_idx], targets[:split_idx]
x_test, y_test = images[split_idx:], targets[split_idx:]

# Normalize the targets to [0, 1]
y_train /= (image_size * image_size)
y_test /= (image_size * image_size)

# Build the model
def build_model():
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1, activation='linear')  # Output a single scalar value
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Instantiate and train the model
model = build_model()
model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, mae = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Predict on test data
predictions = model.predict(x_test[:5])
print(f"Predictions: {predictions.flatten()}, Ground Truth: {y_test[:5]}")
