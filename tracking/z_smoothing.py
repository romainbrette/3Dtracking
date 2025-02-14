'''
Smoothes z by minimizing the speed variation.

Note: this could work also without z estimation (for small depths), although of course there is a symmetry.
'''
import tensorflow as tf
from scipy.ndimage import median_filter
import numpy as np

def smooth_time_series(x, y, z, lambda_smooth=1.0, learning_rate=0.1, iterations=500):
    """
    Smooths the z-coordinate using TensorFlow optimization.

    Args:
        x, y, z: 1D NumPy arrays representing the 3D coordinates.
        lambda_smooth: Weight for smoothness regularization.
        learning_rate: Step size for gradient descent.
        iterations: Number of optimization steps.

    Returns:
        Smoothed z-coordinates.
    """
    z_init = median_filter(z, size=21)
    #z_init = np.random.randn(len(z))*100.
    #z_init = np.linspace(z[0], z[-1], len(z))
    z_orig = tf.constant(z, dtype=tf.float32)
    z_smooth = tf.Variable(z_init, dtype=tf.float32)  # Learnable smoothed z

    x, y = tf.constant(x, dtype=tf.float32), tf.constant(y, dtype=tf.float32)

    # Compute 3D velocity
    def velocity(x, y, z):
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        dz = z[1:] - z[:-1]
        return tf.sqrt(dx**2 + dy**2 + dz**2)

    # Loss function
    def loss():
        vel = velocity(x, y, z_smooth)
        acc = vel[1:] - vel[:-1]  # Acceleration (change in speed)
        data_loss = tf.reduce_mean((z_smooth - z_orig) ** 2)
        smooth_loss = tf.reduce_mean(acc**2)
        print(data_loss, smooth_loss)
        return data_loss + lambda_smooth * smooth_loss

    # Optimize
    optimizer = tf.optimizers.Adam(learning_rate)
    for _ in range(iterations):
        #print(loss())
        optimizer.minimize(loss, var_list=[z_smooth])

    return z_smooth.numpy()
