#!/usr/bin/env python3

from adafruit_ads1x15.analog_in import AnalogIn
import numpy as np

class ServoMotor:
    def __init__(self, analog_pin: AnalogIn, pwm_index: int, meas_pwm_diff: float, volt_to_deg: float, zero_deg_voltage: float, inverted: bool = False, ros_rate: int = 120):
        self._analog_pin = analog_pin
        self._pwm_index = pwm_index
        self._meas_pwm_diff = meas_pwm_diff
        self._volt_to_deg = volt_to_deg
        self._zero_deg_voltage = zero_deg_voltage
        self._inverted = inverted

        # Measurements
        self.filtered_measured_angle = self.get_angle_measurement() # initialise with first measurement
        self._old_filtered_measured_angle = self.filtered_measured_angle # to calculate velocity
        self._measurement_buffer = np.array([self.filtered_measured_angle]*2)
        self.filtered_velocity = 0.0 
        self._velocity_buffer = np.array([self.filtered_velocity]*2)
        self._measurement_threshold = 80

        # Target
        self.target_angle = self.filtered_measured_angle
        self._target_velocity = 0

        # Guidance
        self._c = 1
        self._w = 30
        self._max_velocity = 300
        self._dt = 1/ros_rate

    def set_guidance_params(self, c: float = 1, w: float = 30, max_velocity: float = 300):
        self._c = c
        self._w = w
        self._max_velocity = max_velocity

    def low_pass_guidance_second_order_module(self, reference: float):
        c = self._c
        w = self._w
        
        if np.abs(self._target_velocity) > self._max_velocity:
            motor_targets_dot = self._max_velocity * np.sign(self._target_velocity)
            self.target_angle = self.target_angle + motor_targets_dot * self._dt
            self._target_velocity = motor_targets_dot
        else:
            motor_targets_dot = self._target_velocity
            vel_dot = -2*c*w*self._target_velocity + w**2*(reference - self.target_angle)
            self.target_angle = self.target_angle + motor_targets_dot * self._dt
            self._target_velocity = self._target_velocity + vel_dot * self._dt
    
    def get_angle_measurement(self):
        if self._inverted:
            return -((self._analog_pin.voltage  - self._zero_deg_voltage) * self._volt_to_deg - 173.5 )+ self._meas_pwm_diff
        else:
            return (self._analog_pin.voltage  - self._zero_deg_voltage) * self._volt_to_deg + self._meas_pwm_diff
        
    def update_filtered_measurement(self):
        self._old_filtered_measured_angle = self.filtered_measured_angle
        new_measurement = self.get_angle_measurement()
        if abs(new_measurement - self._measurement_buffer[-1]) < self._measurement_threshold:
            self._measurement_buffer = np.roll(self._measurement_buffer, -1)
            self._measurement_buffer[-1] = new_measurement
            self.filtered_measured_angle = np.mean(self._measurement_buffer)

    def update_filtered_velocity(self):
        self._velocity_buffer = np.roll(self._velocity_buffer, -1)
        self._velocity_buffer[-1] = (self.filtered_measured_angle - self._old_filtered_measured_angle) / self._dt
        self.filtered_velocity = np.mean(self._velocity_buffer)