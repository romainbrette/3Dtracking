'''
Augmentation
'''
import tensorflow as tf

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
