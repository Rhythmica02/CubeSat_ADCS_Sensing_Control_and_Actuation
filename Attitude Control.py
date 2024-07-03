#Import necessary Libraries
import numpy as np
import matplotlib.pyplot as plt

#Defining Functions
class AttitudeDeterminationSystem:
    def __init__(self):
        # Initialize sensor data
        self.magnetometer_data = None
        self.imu_data = None
        self.sun_sensor_data = None
        self.gps_data = None
        
        # Initialize attitude variables
        self.attitude_quaternion = np.array([1, 0, 0, 0])  

    def update_sensor_data(self, magnetometer, imu, sun_sensor, gps):
        self.magnetometer_data = magnetometer
        self.imu_data = imu
        self.sun_sensor_data = sun_sensor
        self.gps_data = gps
        
    def estimate_attitude(self):
        # Perform sensor-based attitude estimation
        mag_attitude = self.estimate_magnetometer_attitude()
        imu_attitude = self.estimate_imu_attitude()
        sun_attitude = self.estimate_sun_sensor_attitude()
        fused_attitude = self.sensor_fusion(mag_attitude, imu_attitude, sun_attitude)
        final_attitude = self.point_towards_earth(fused_attitude)
        return final_attitude
    
    def estimate_magnetometer_attitude(self):
        # Magnetometer-based attitude estimation algorithm
        mag_data = self.magnetometer_data
        # Assuming magnetometer data is in the form [x, y, z]
        roll = np.arctan2(mag_data[1], mag_data[0])
        pitch = np.arctan2(-mag_data[2], np.sqrt(mag_data[0]**2 + mag_data[1]**2))
        yaw = 0  
        return self.quaternion_from_euler(roll, pitch, yaw)
    
    def estimate_imu_attitude(self):
        # IMU-based attitude estimation algorithm 
        imu_data = self.imu_data
        dt = 1  
        gyro_data = imu_data  
        roll_rate, pitch_rate, yaw_rate = gyro_data
        roll, pitch, yaw = self.euler_from_quaternion(self.attitude_quaternion)
        roll += roll_rate * dt
        pitch += pitch_rate * dt
        yaw += yaw_rate * dt
        return self.quaternion_from_euler(roll, pitch, yaw)
    
    def estimate_sun_sensor_attitude(self):
        # Sun sensor-based attitude estimation algorithm
        sun_data = self.sun_sensor_data
        # Assuming sun sensor data is in the form [azimuth, elevation]
        azimuth, elevation = sun_data
        # Convert sun angles to quaternion
        # Assuming sun is aligned with positive z-axis in the CubeSat frame
        roll = 0
        pitch = -elevation
        yaw = azimuth
        return self.quaternion_from_euler(roll, pitch, yaw)
    
    def sensor_fusion(self, mag_attitude, imu_attitude, sun_attitude):
        # Sensor fusion algorithm
        fused_attitude = (mag_attitude + imu_attitude + sun_attitude) / 3
        return fused_attitude / np.linalg.norm(fused_attitude)
    
    def point_towards_earth(self, attitude_quaternion):
        earth_vector = np.array([0, 0, 1])  
        rotated_earth_vector = self.rotate_vector(attitude_quaternion, earth_vector)
        earth_attitude = self.quaternion_from_two_vectors(rotated_earth_vector, np.array([0, 0, 1]))
        return self.quaternion_multiply(attitude_quaternion, earth_attitude)[:3]
    
    def rotate_vector(self, quaternion, vector):
        # Rotate a vector by a quaternion
        q_conj = np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])
        return self.quaternion_multiply(self.quaternion_multiply(quaternion, vector), q_conj)[1:]
    
    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1 if len(q1) == 4 else (0, q1[0], q1[1], q1[2])
        w2, x2, y2, z2 = q2 if len(q2) == 4 else (0, q2[0], q2[1], q2[2])
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z])
    
    def quaternion_from_euler(self, roll, pitch, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        qw = cy * cp * cr + sy * sp * sr
        qx = cy * cp * sr - sy * sp * cr
        qy = sy * cp * sr + cy * sp * cr
        qz = sy * cp * cr - cy * sp * sr

        return np.array([qw, qx, qy, qz])
    
    def euler_from_quaternion(self, quaternion):
        if len(quaternion) == 3:
            quaternion = np.concatenate(([1], quaternion))
        qw, qx, qy, qz = quaternion
        roll = np.arctan2(2 * (qw*qx + qy*qz), 1 - 2 * (qx**2 + qy**2))
        pitch = np.arcsin(2 * (qw*qy - qz*qx))
        yaw = np.arctan2(2 * (qw*qz + qx*qy), 1 - 2 * (qy**2 + qz**2))
        return roll, pitch, yaw
    
    def quaternion_from_two_vectors(self, u, v):
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0 or norm_v == 0:
            return np.array([1, 0, 0, 0])
        else:
            dot_product = np.dot(u, v)
            cos_theta = dot_product / (norm_u * norm_v)
            half_theta = np.arccos(np.clip(cos_theta, -1, 1)) / 2
            axis = np.cross(u, v)
            axis /= np.linalg.norm(axis)
            return self.quaternion_from_axis_angle(axis, 2 * half_theta)

    def quaternion_from_axis_angle(self, axis, angle):
        half_angle = angle / 2
        sin_half_angle = np.sin(half_angle)
        cos_half_angle = np.cos(half_angle)
        return np.array([cos_half_angle, sin_half_angle * axis[0], sin_half_angle * axis[1], sin_half_angle * axis[2]])

    def plot_attitude(self, attitude_quaternion):
        roll_data = []
        pitch_data = []
        yaw_data = []
        time = np.arange(0, 10, 0.1) 
        
        for t in time:
            roll, pitch, yaw = self.euler_from_quaternion(attitude_quaternion)
            roll_data.append(roll)
            pitch_data.append(pitch)
            yaw_data.append(yaw)
            
            attitude_quaternion = self.update_quaternion(attitude_quaternion)
        
        plt.figure(figsize=(10, 6))
        plt.plot(time, roll_data, label='Roll')
        plt.plot(time, pitch_data, label='Pitch')
        plt.plot(time, yaw_data, label='Yaw')
        plt.title('Attitude over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (rad)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def update_quaternion(self, quaternion):
        return quaternion  

    def update_quaternion(self, quaternion):
        delta_roll = 0.01
        delta_pitch = 0.02
        delta_yaw = 0.03
    
        roll, pitch, yaw = self.euler_from_quaternion(quaternion)
        new_roll = roll + delta_roll
        new_pitch = pitch + delta_pitch
        new_yaw = yaw + delta_yaw
    
        return self.quaternion_from_euler(new_roll, new_pitch, new_yaw)


if __name__ == "__main__":
    adc_system = AttitudeDeterminationSystem()
    
    magnetometer_data = np.array([0.1, 0.2, 0.3])  
    imu_data = np.array([0.1, 0.2, 0.3]) 
    sun_sensor_data = np.array([0.1, 0.2])  
    gps_data = np.array([0.1, 0.2, 0.3])  
    
    # Update sensor data
    adc_system.update_sensor_data(magnetometer_data, imu_data, sun_sensor_data, gps_data)
    
    # Perform attitude determination
    estimated_attitude = adc_system.estimate_attitude()
    
    print("Estimated Attitude (Quaternion):", estimated_attitude)
    
    # Plot attitude over time
    adc_system.plot_attitude(estimated_attitude)
