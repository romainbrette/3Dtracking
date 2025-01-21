import tensorflow as tf
import matplotlib.pyplot as plt

# Function to visualize a batch of images
def visualize_dataset(dataset, num_images=9):
    """Visualize a few images from the dataset."""

    # Take one batch of data
    for i, (image, label) in enumerate(dataset.take(5)):
        plt.figure(figsize=(10, 10))
        # If batch size > num_images, only display num_images
        for j in range(min(num_images, image.shape[0])):
            plt.subplot(3, 3, j + 1)  # 3x3 grid for 9 images
            plt.imshow(tf.squeeze(image[j]), cmap="gray")  # Use cmap="gray" for grayscale
            plt.title(f"Label: {label[j].numpy()}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()
