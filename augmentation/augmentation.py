'''
Augmentation
'''
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

class IntensityNormalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        mean_intensity = tf.reduce_mean(inputs)
        mean_intensity = tf.maximum(mean_intensity, 1e-8)
        return inputs / mean_intensity

class RandomThreshold(tf.keras.layers.Layer):
    def __init__(self, min=0., max=.1, **kwargs):
        super().__init__(**kwargs)
        self.min = min
        self.max = max

    def call(self, inputs):
        threshold = tf.random.uniform([], self.min, self.max)
        return tf.clip_by_value(inputs, threshold, 1e6) - threshold

    def get_config(self):
        config = super().get_config()
        config.update({
            "min": self.min,
            "max": self.max,
        })
        return config

class RandomIntensityScaling(tf.keras.layers.Layer): ## doesn't this do the same scaling in all the batch?
    def __init__(self, min_scale=0.8, max_scale=1.2, **kwargs):
        super().__init__(**kwargs)
        self.min_scale = min_scale
        self.max_scale = max_scale

    def call(self, inputs):

        scale = tf.random.uniform([], self.min_scale, self.max_scale)
        scaled_image = inputs * scale
        return scaled_image
        #return tf.clip_by_value(scaled_image, 0.0, 1.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
        })
        return config

class RandomOcclusion(tf.keras.layers.Layer):
    def __init__(self, black_background=True, **kwargs):
        super().__init__(**kwargs)
        self.black_background = black_background

    def call(self, inputs):
        pass

    def get_config(self):
        config = super().get_config()
        config.update({
        })
        return config

# Define a custom rotation function
def random_rotate(image, max_angle=180):
    """Rotate the image by a random angle between -max_angle and max_angle."""
    # Convert max_angle to radians
    max_angle_rad = max_angle * np.pi / 180.0
    # Generate a random angle
    angle = tf.random.uniform([], -max_angle_rad, max_angle_rad)
    # Rotate the image
    rotated_image = tfa.image.rotate(image, angle, interpolation='BILINEAR')
    return rotated_image

def normalize_intensity(image):
    mean_intensity = tf.reduce_mean(image)
    mean_intensity = tf.maximum(mean_intensity, 1e-8)
    return image / mean_intensity

def random_threshold(image, max_threshold):
    threshold = tf.random.uniform([], max_threshold)
    return tf.clip_by_value(image, threshold, 1e6) - threshold

def random_intensity(image, min_scale=0.8, max_scale=1.2):
    scale = tf.random.uniform([], min_scale, max_scale)
    scaled_image = image * scale
    return scaled_image

def add_noise(image, sigma=1.):
    scale = tf.random.uniform([], 0, 1)
    return image+tf.random.normal(shape=tf.shape(image), mean=0., stddev=sigma, dtype=image.dtype)*scale
