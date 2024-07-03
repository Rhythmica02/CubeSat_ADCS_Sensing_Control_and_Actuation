#Importing necessary Libraries
import numpy as np
import matplotlib.pyplot as plt

#Defining Functions
class Sensor:
    def __init__(self, name):
        self.name = name

    def read_data(self):
        # Simulated data reading
        return np.random.uniform(0, 1)  

class Actuator:
    def __init__(self, name):
        self.name = name
        self.actuator_data = []  

    def actuate(self, command):
        print(f"{self.name} Actuator Command: {command}")
        self.actuator_data.append(command)  

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0

    def calculate(self, current_value):
        error = self.setpoint - current_value
        self.integral += error
        derivative = error - self.prev_error

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        self.prev_error = error

        return output

# Initialize sensors, actuators, and controller
sun_sensor = Sensor("Sun Sensor")
gyroscope = Sensor("Gyroscope")
magnetometer = Sensor("Magnetometer")
actuator = Actuator("Actuator")
pid_controller = PIDController(0.1, 0.01, 0.05, 0)  

# Simulation loop
sensor_data = {'Sun Sensor': [], 'Gyroscope': [], 'Magnetometer': []}
actuator_commands = []
for _ in range(100):
    sun_data = sun_sensor.read_data()
    gyro_data = gyroscope.read_data()
    mag_data = magnetometer.read_data()

    sensor_data['Sun Sensor'].append(sun_data)
    sensor_data['Gyroscope'].append(gyro_data)
    sensor_data['Magnetometer'].append(mag_data)

    # Calculate control output
    control_output = pid_controller.calculate(sun_data)  

    actuator.actuate(control_output)
    actuator_commands.append(control_output)

# Plot sensor and actuator data
plt.figure(figsize=(12, 6))
plt.plot(sensor_data['Sun Sensor'], label='Sun Sensor Data')
plt.plot(sensor_data['Gyroscope'], label='Gyroscope Data')
plt.plot(sensor_data['Magnetometer'], label='Magnetometer Data')
plt.plot(actuator.actuator_data, label='Actuator Data', linestyle='--')  
plt.legend()
plt.xlabel('Time')
plt.ylabel('Sensor Output / Actuator Command')
plt.title('Sensor and Actuator Data Plot')
plt.show()
