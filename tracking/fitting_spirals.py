'''
I fit spiraling trajectories seen in 2D with a propulsion kinematic model.
https://chatgpt.com/canvas/shared/67cac8fea43c8191b2fdc36316f5c323
'''
import numpy as np
import tensorflow as tf


# Define the skew-symmetric matrix for angular velocity
def skew_symmetric(omega):
    omega_x, omega_y, omega_z = tf.unstack(omega, axis=-1)
    return tf.stack([
        [tf.zeros_like(omega_x), -omega_z, omega_y],
        [omega_z, tf.zeros_like(omega_y), -omega_x],
        [-omega_y, omega_x, tf.zeros_like(omega_z)]
    ], axis=-1)  # Shape (batch, 3, 3)


# Convert angle and speed to angular velocity vector
def angular_velocity(axis_angle, omega_magnitude):
    axis = tf.stack([
        tf.cos(axis_angle), tf.sin(axis_angle), tf.zeros_like(axis_angle)
    ], axis=-1)  # Constraining to XY plane
    return axis * omega_magnitude[..., None]  # Scale by magnitude


# Define the trajectory computation using TensorFlow
@tf.function
def compute_trajectory(v_body, axis_angle, omega_magnitude, x0, R0, dt, steps):
    x = x0
    R = R0
    positions = [x]

    for i in range(steps):
        omega_body = angular_velocity(axis_angle[i], omega_magnitude[i])
        v_world = tf.linalg.matvec(R, v_body[i])  # Convert velocity to world frame
        R_dot = tf.linalg.matmul(R, skew_symmetric(omega_body))  # Rotation matrix derivative

        x = x + v_world * dt
        R = R + R_dot * dt  # Euler integration

        positions.append(x)

    return tf.stack(positions)


# Initial conditions
x0 = tf.constant([0.0, 0.0, 0.0])  # Initial position
R0 = tf.eye(3)  # Initial orientation

# Simulation parameters
dt = 0.1
steps = 100

# Trainable variables: motion along z-axis only, axis angle, and rotation speed
v_body = tf.Variable(tf.random.normal((steps, 1)), dtype=tf.float32)  # Only z-component is trainable
v_body = tf.concat([tf.zeros((steps, 2)), v_body], axis=-1)  # Force motion along z-axis

axis_angle = tf.Variable(tf.random.normal((steps,)), dtype=tf.float32)  # Rotation axis angle in XY plane
omega_magnitude = tf.Variable(tf.random.normal((steps,)), dtype=tf.float32)  # Rotation speed

# Load measured 2D trajectory (x, y)
measured_xy = tf.placeholder(tf.float32, shape=(steps, 2))

# Compute trajectory
trajectory = compute_trajectory(v_body, axis_angle, omega_magnitude, x0, R0, dt, steps)
predicted_xy = trajectory[:, :2]  # Extract (x, y) components

# Define loss function
empirical_error = tf.reduce_mean(tf.square(predicted_xy - measured_xy))
smoothness_v = tf.reduce_mean(tf.square(v_body[1:] - v_body[:-1]))
smoothness_w = tf.reduce_mean(tf.square(omega_magnitude[1:] - omega_magnitude[:-1]))
smoothness_a = tf.reduce_mean(tf.square(axis_angle[1:] - axis_angle[:-1]))

loss = empirical_error + 0.1 * (smoothness_v + smoothness_w + smoothness_a)  # Weighting smoothness

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


# Training step
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        loss_value = loss
    grads = tape.gradient(loss_value, [v_body, axis_angle, omega_magnitude])
    optimizer.apply_gradients(zip(grads, [v_body, axis_angle, omega_magnitude]))
    return loss_value


# Training loop
for i in range(1000):  # Adjust iterations as needed
    loss_val = train_step()
    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss_val.numpy()}")

# Print optimized final position
print("Optimized Final Position:", trajectory[-1].numpy())
