#Importing necessary Libraries
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

Kp = 0.1  # Proportional gain
Ki = 0.01  # Integral gain
Kd = 0.05  # Derivative gain
dt = 0.1  # Time step (in seconds)
max_torque_rw = 0.02  # Maximum torque per reaction wheel (in N.m)
max_momentum_rw = 0.005  # Maximum reaction wheel momentum (in N.m.s)
max_torque_mtq = 0.001  # Maximum torque per magnetorquer (in N.m)
J = np.diag([0.01, 0.02, 0.03])  # Satellite inertia tensor (in kg.m^2)
alt = 450  # Altitude (in km)
inc = np.deg2rad(96.5)  # Inclination (in radians)
ecc = 0.0594  # Eccentricity
semi_major_axis = 7153.14  # Semi-major axis (in km)
period = 6020.8  # Orbital period (in seconds)
simulation_time = 10  # Total simulation time (in seconds)

# Initial conditions
q_des = np.array([1, 0, 0, 0])  # Desired quaternion (no rotation)
q_curr = np.array([0.9, 0.1, 0.2, 0.3])  # Current quaternion
w_curr = np.array([0.001, 0.002, -0.003])  # Current angular velocity (in rad/s)
torque_cmd_rw = np.zeros(3)  # Initial reaction wheel torque command (in N.m)
torque_cmd_mtq = np.zeros(3)  # Initial magnetorquer torque command (in N.m)
integral_error = np.zeros(3)  # Integral error (for integral control)
q_err_prev = np.zeros(3)  # Initialize previous quaternion error
position = np.array([0.0, 0.0, 0.0])  # Initialize position (placeholder)
time = 0.0  # Initialize time (placeholder)

# Lists to store data for plotting
q_history = [q_curr]
w_history = [w_curr]
torque_rw_history = [torque_cmd_rw]
torque_mtq_history = [torque_cmd_mtq]
time_history = [time]

# Function to compute quaternion error
def quaternion_error(q1, q2):
    q_err = np.array([q2[0], -q2[1], -q2[2], -q2[3]]) * q1
    q_err = q_err[1:]  # Drop the scalar part
    return q_err

# Function to compute Earth's magnetic field (using IGRF model)
def magnetic_field(position, time):
   
    mag_field = np.array([0.2, 0.1, -0.3])  # Earth's magnetic field vector (in Gauss)
    return mag_field

# Main control loop
while time < simulation_time:
    # Compute quaternion error
    q_err = quaternion_error(q_curr, q_des)

    # PID control
    proportional_term = Kp * q_err
    integral_error += q_err * dt
    integral_term = Ki * integral_error
    derivative_term = Kd * (q_err - q_err_prev) / dt
    torque_cmd = proportional_term + integral_term + derivative_term

    # Saturate reaction wheel torque command
    torque_cmd_rw = np.clip(torque_cmd, -max_torque_rw, max_torque_rw)

    # Compute magnetorquer torque command
    mag_field = magnetic_field(position, time)
    torque_cmd_mtq = np.cross(mag_field, J @ w_curr) - torque_cmd_rw

    # Saturate magnetorquer torque command
    torque_cmd_mtq = np.clip(torque_cmd_mtq, -max_torque_mtq, max_torque_mtq)

    # Update angular velocity (simplified dynamics)
    w_curr += (torque_cmd_rw + torque_cmd_mtq) * dt / np.diag(J)

    # Update quaternion (simplified kinematics)
    q_curr += 0.5 * np.concatenate([[0], w_curr]) * q_curr * dt

    # Normalize quaternion
    q_curr /= np.linalg.norm(q_curr)

    # Update previous error
    q_err_prev = q_err

    
    time += dt

    # Store data for plotting
    q_history.append(q_curr)
    w_history.append(w_curr)
    torque_rw_history.append(torque_cmd_rw)
    torque_mtq_history.append(torque_cmd_mtq)
    time_history.append(time)

    # Print current state (for debugging)
    print(f"Time: {time:.2f} s")
    print(f"Current quaternion: {q_curr}")
    print(f"Current angular velocity: {w_curr}")
    print(f"Reaction wheel torque command: {torque_cmd_rw}")
    print(f"Magnetorquer torque command: {torque_cmd_mtq}")
    print()

# Plotting
q_history = np.array(q_history)
w_history = np.array(w_history)
torque_rw_history = np.array(torque_rw_history)
torque_mtq_history = np.array(torque_mtq_history)
time_history = np.array(time_history)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].plot(time_history, q_history[:, 0], label='q0')
axs[0, 0].plot(time_history, q_history[:, 1], label='q1')
axs[0, 0].plot(time_history, q_history[:, 2], label='q2')
axs[0, 0].plot(time_history, q_history[:, 3], label='q3')
axs[0, 0].set_title('Quaternion')
axs[0, 0].legend()

axs[0, 1].plot(time_history, w_history[:, 0], label='wx')
axs[0, 1].plot(time_history, w_history[:, 1], label='wy')
axs[0, 1].plot(time_history, w_history[:, 2], label='wz')
axs[0, 1].set_title('Angular Velocity')
axs[0, 1].legend()

axs[1, 0].plot(time_history, torque_rw_history[:, 0], label='tx')
axs[1, 0].plot(time_history, torque_rw_history[:, 1], label='ty')
axs[1, 0].plot(time_history, torque_rw_history[:, 2], label='tz')
axs[1, 0].set_title('Reaction Wheel Torque Command')
axs[1, 0].legend()

axs[1, 1].plot(time_history, torque_mtq_history[:, 0], label='tx')
axs[1, 1].plot(time_history, torque_mtq_history[:, 1], label='ty')
axs[1, 1].plot(time_history, torque_mtq_history[:, 2], label='tz')
axs[1, 1].set_title('Magnetorquer Torque Command')
axs[1, 1].legend()

plt.tight_layout()
plt.show()
