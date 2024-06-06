#!/usr/bin/env python3

# ROS imports
import rospy
import rospkg
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Quaternion, Pose, PoseStamped

# Hardware imports
import board 
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.ads1x15 import Mode
from adafruit_ads1x15.analog_in import AnalogIn
import adafruit_tca9548a
from utils.servo_driver import PCA9685

# Custom imports
from utils.servo_motor import ServoMotor
from utils.utils import quat_from_euler_xyz, quat_conjugate, quat_mul, quat_diff_rad, ang_vel_from_quat
from utils.policy import EurepusPolicy

# Python imports
import json
from typing import List
import signal
import sys
import select
import numpy as np
import torch
import onnx
import onnxruntime as ort
import time

# from pynput import keyboard
# import asyncio
from sshkeyboard import listen_keyboard, stop_listening


class Eurepus:
    def __init__(self):
        # Init ROS things
        rospy.init_node('eurepus')
        rospack = rospkg.RosPack()

        # Cfg
        self._use_laterals = True
        self._pitch_configuration =  True
        self._use_guidance_module = False
        self._policy_rate = 80
        self._measurement_rate = 120
        self._interpol_coeff = 0
        self._guidance_max_velocity = 1000
        self._pth_file_path = rospack.get_path('eurepus_control') + "/models/eurepus_070.pth"
        self._log_data = True
        self._data_save_path = rospack.get_path('eurepus_control') + "/data/drop_test_070_roll_-90_deg_120hz_measurement_01.txt"
        self._ros_rate = 240 # Hz
        self._max_time = 1.5 #s
        self._lateral_sim_limits = [-50, 90]
        self._transversal_sim_limits = [-20, 140]
        self._max_transversal_motor_diff = -1
        self._max_transversal_motor_sum = 220
        
        # Leg configuration
        with open(rospack.get_path('eurepus_control') + '/scripts/config.json') as f:
            self.cfg = json.load(f)

        self.load_cfg_vals()

        # Create the I2C bus
        self._i2c = busio.I2C(board.GPIOX_18, board.GPIOX_17)

        # Create the Multiplexer object
        self._tca = adafruit_tca9548a.TCA9548A(self._i2c)

        # Create ADCs and configure them
        self._ADCs_BL = [ADS.ADS1115(self._tca[3]), ADS.ADS1115(self._tca[3], address=0x4a), ADS.ADS1115(self._tca[3], address=0x49)]
        self._ADCs_FL = [ADS.ADS1115(self._tca[2]), ADS.ADS1115(self._tca[2], address=0x4a), ADS.ADS1115(self._tca[2], address=0x49)]
        self._ADCs_BR = [ADS.ADS1115(self._tca[0]), ADS.ADS1115(self._tca[0], address=0x4a), ADS.ADS1115(self._tca[0], address=0x49)]
        self._ADCs_FR = [ADS.ADS1115(self._tca[1]), ADS.ADS1115(self._tca[1], address=0x4a), ADS.ADS1115(self._tca[1], address=0x49)]

        for adc in self._ADCs_BL + self._ADCs_FL + self._ADCs_BR + self._ADCs_FR:
            adc.mode = Mode.CONTINUOUS
            adc.data_rate = 860

        # Create the legs objects [lateral, inner transversal, outer transversal]
        self._legs_BL = [
            ServoMotor(AnalogIn(self._ADCs_BL[0], ADS.P2), 9, self.LateralMotor_BL_meas_pwm_diff, self.LateralMotor_BL_volt_to_deg, self.LateralMotor_BL_zero_deg_voltage, self.LateralMotor_BL_inverted, self._measurement_rate),
            ServoMotor(AnalogIn(self._ADCs_BL[1], ADS.P2), 10, self.InnerTransversalMotor_BL_meas_pwm_diff, self.InnerTransversalMotor_BL_volt_to_deg, self.InnerTransversalMotor_BL_zero_deg_voltage, self.InnerTransversalMotor_BL_inverted, self._measurement_rate),
            ServoMotor(AnalogIn(self._ADCs_BL[2], ADS.P2), 11, self.OuterTransversalMotor_BL_meas_pwm_diff, self.OuterTransversalMotor_BL_volt_to_deg, self.OuterTransversalMotor_BL_zero_deg_voltage, self.OuterTransversalMotor_BL_inverted, self._measurement_rate)
        ]

        self._legs_FL = [
            ServoMotor(AnalogIn(self._ADCs_FL[0], ADS.P2), 0, self.LateralMotor_FL_meas_pwm_diff, self.LateralMotor_FL_volt_to_deg, self.LateralMotor_FL_zero_deg_voltage, self.LateralMotor_FL_inverted, self._measurement_rate),
            ServoMotor(AnalogIn(self._ADCs_FL[1], ADS.P2), 1, self.InnerTransversalMotor_FL_meas_pwm_diff, self.InnerTransversalMotor_FL_volt_to_deg, self.InnerTransversalMotor_FL_zero_deg_voltage, self.InnerTransversalMotor_FL_inverted, self._measurement_rate),
            ServoMotor(AnalogIn(self._ADCs_FL[2], ADS.P2), 2, self.OuterTransversalMotor_FL_meas_pwm_diff, self.OuterTransversalMotor_FL_volt_to_deg, self.OuterTransversalMotor_FL_zero_deg_voltage, self.OuterTransversalMotor_FL_inverted, self._measurement_rate)
        ]

        self._legs_BR = [
            ServoMotor(AnalogIn(self._ADCs_BR[0], ADS.P2), 6, self.LateralMotor_BR_meas_pwm_diff, self.LateralMotor_BR_volt_to_deg, self.LateralMotor_BR_zero_deg_voltage, self.LateralMotor_BR_inverted, self._measurement_rate),
            ServoMotor(AnalogIn(self._ADCs_BR[1], ADS.P2), 7, self.InnerTransversalMotor_BR_meas_pwm_diff, self.InnerTransversalMotor_BR_volt_to_deg, self.InnerTransversalMotor_BR_zero_deg_voltage, self.InnerTransversalMotor_BR_inverted, self._measurement_rate),
            ServoMotor(AnalogIn(self._ADCs_BR[2], ADS.P2), 8, self.OuterTransversalMotor_BR_meas_pwm_diff, self.OuterTransversalMotor_BR_volt_to_deg, self.OuterTransversalMotor_BR_zero_deg_voltage, self.OuterTransversalMotor_BR_inverted, self._measurement_rate)
        ]

        self._legs_FR = [
            ServoMotor(AnalogIn(self._ADCs_FR[0], ADS.P0), 3, self.LateralMotor_FR_meas_pwm_diff, self.LateralMotor_FR_volt_to_deg, self.LateralMotor_FR_zero_deg_voltage, self.LateralMotor_FR_inverted, self._measurement_rate),
            ServoMotor(AnalogIn(self._ADCs_FR[1], ADS.P0), 4, self.InnerTransversalMotor_FR_meas_pwm_diff, self.InnerTransversalMotor_FR_volt_to_deg, self.InnerTransversalMotor_FR_zero_deg_voltage, self.InnerTransversalMotor_FR_inverted, self._measurement_rate),
            ServoMotor(AnalogIn(self._ADCs_FR[2], ADS.P0), 5, self.OuterTransversalMotor_FR_meas_pwm_diff, self.OuterTransversalMotor_FR_volt_to_deg, self.OuterTransversalMotor_FR_zero_deg_voltage, self.OuterTransversalMotor_FR_inverted, self._measurement_rate)
        ]

        self._legs = self._legs_BL + self._legs_FL + self._legs_BR + self._legs_FR

        self._legs_to_output_idx = [9, 10, 11, 0, 1, 2, 6, 7, 8, 3, 4, 5]

        for leg in self._legs:
            leg.set_guidance_params(max_velocity=self._guidance_max_velocity)

        # Initialise motor channels
        self._pca = PCA9685(self._tca[7], reference_clock_speed=25000000)
        self._pca.frequency = 300 # Gives ~327Hz PWM frequency, measured with oscilloscope
    
        self._motor_channels = self._pca.channels
        self._motor_channels.set_pulse_width_range(500, 2500)
        self._motor_channels.set_actuation_range(173.5)

        self._motor_angles = [60]*16 # Default angles

        # Initialise eurepus state variables
        self._pitch = 0
        self._base_velocity = 0
        self._old_pitch = 0
        self._base_vel_buffer = np.array([0.0]*2)
        self._base_pos_buffer = np.array([0.0]*2)
        self._base_pos_filtered = 0
        self._base_vel_filtered = 0
        self.old_orient_err = np.array([1, 0, 0, 0])
        self.base_ang_vel = np.array([0, 0, 0])
        self._motor_sim_limits = [self._lateral_sim_limits, self._transversal_sim_limits, self._transversal_sim_limits] # helper variable

        # Initialise policy
        if self._use_laterals:
            num_observations = 31
            num_actions = 12
        else:
            num_observations = 18
            num_actions = 8

        self.policy = EurepusPolicy(num_observations, [128, 64, 64], num_actions, pth_file=self._pth_file_path)

        torch.onnx.export(self.policy,                  # model being run
                  torch.randn(num_observations),        # model input (or a tuple for multiple inputs)
                  rospack.get_path('eurepus_control') + "/scripts/eurepus_policy.onnx",# where to save the model (can be a file or file-like object)
                  input_names = ['input'],              # the model's input names
                  output_names = ['output'])            # the model's output namesd

        onnx_model = onnx.load(rospack.get_path('eurepus_control') + "/scripts/eurepus_policy.onnx")
        onnx.checker.check_model(onnx_model)
        self.ort_sess = ort.InferenceSession(rospack.get_path('eurepus_control') + "/scripts/eurepus_policy.onnx")

        # Subscriber to encoder topic
        self._base_rotation = quat_from_euler_xyz(0, 0, 0)
        self._base_position = np.array([0, 0, 0])
        self.encoder_subscriber = rospy.Subscriber("encoder_value", Float32, self.encoder_callback)

        # Subscriber to reference topic
        self._potmeter_ref = 0
        self.reference_subscriber = rospy.Subscriber("reference", Float32, self.reference_callback)

        self.mocap_subscriber = rospy.Subscriber("/qualisys/eurepus/pose", PoseStamped , self.mocap_callback)

        self._robot_initialised = False
        self._robot_falling = False

        self.counter = 0
        self.output = [0]*12
        self.timeout = False

        references = [0, 0, -40, 40, -80, 80, -20, 20, 0, 0, 0, 0, 0, 0, 0]
        self.pole_ref = 0

        rospy.loginfo("Eurepus Motor node initialised")

    def load_cfg_vals(self):
        # Iterate through each motor entry in cfg and create member variables
        for motor_id, motor_data in self.cfg["motors"].items():
            # Create member variables using setattr
            for key, value in motor_data.items():
                setattr(self, motor_id + "_" + key, value)
    
    def encoder_callback(self, msg):
        if self._pitch_configuration: self._base_pos = msg.data * np.pi/180
        else: self._base_pos = - msg.data * np.pi/180

        self._base_rotation = quat_from_euler_xyz( 0, self._base_pos, 0)

        self._base_velocity = (self._pitch - self._old_pitch) * self._ros_rate
        self._old_pitch = self._pitch

        self._base_pos_buffer = np.roll(self._base_pos_buffer, -1)
        self._base_pos_buffer[1] = self._base_pos
        self._base_pos_filtered = np.mean(self._base_pos_buffer)

        self._base_vel_buffer = np.roll(self._base_vel_buffer, -1)
        self._base_vel_buffer[1] = self._base_velocity
        self._base_vel_filtered = np.mean(self._base_vel_buffer)

    def mocap_callback(self, msg):
        if not self._robot_initialised:
            self._base_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            return
        
        if not self._robot_falling:
            new_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            if np.abs(new_position[2] - self._base_position[2]) > 0.03:
                self._robot_falling = True
                self.policy_start_time = rospy.get_time()
            else:
                return

        if self.counter%(self._ros_rate/self._measurement_rate) == 0:
            self._base_rotation = np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])

            ###################################################################
            ## Read and filter analog position signals
            ###################################################################

            for leg in self._legs:
                leg.update_filtered_measurement() # reads new measurement and updates filtered angle
                leg.update_filtered_velocity() # uses the last two filtered angles to calculate velocity


        if self.counter%(self._ros_rate/self._policy_rate) == 0:
            start_time = rospy.get_time()
            
            ####################################################################
            # Compute actions
            ####################################################################
            
            # Load reference
            orient_ref = quat_from_euler_xyz(-90 * np.pi/180, 0, 0)
            orient_err = quat_mul(quat_conjugate(orient_ref), self._base_rotation )
            self.base_ang_vel = ang_vel_from_quat(self.old_orient_err, orient_err, 1/self._policy_rate)
            self.old_orient_err = orient_err

            # if orient_err[0] < 0:
            #     orient_err = -orient_err

            obs = np.array([
                            (self._legs_BL[0].filtered_measured_angle + self.LateralMotor_BL_sim2real_offset)          * np.pi/180 ,
                            (self._legs_BR[0].filtered_measured_angle + self.LateralMotor_BR_sim2real_offset)          * np.pi/180 ,
                            (self._legs_FL[0].filtered_measured_angle + self.LateralMotor_FL_sim2real_offset)          * np.pi/180 ,
                            (self._legs_FR[0].filtered_measured_angle + self.LateralMotor_FR_sim2real_offset)          * np.pi/180 ,
                            (self._legs_BL[2].filtered_measured_angle + self.OuterTransversalMotor_BL_sim2real_offset) * np.pi/180,
                            (self._legs_BL[1].filtered_measured_angle + self.InnerTransversalMotor_BL_sim2real_offset) * np.pi/180,
                            (self._legs_BR[2].filtered_measured_angle + self.OuterTransversalMotor_BR_sim2real_offset) * np.pi/180,
                            (self._legs_BR[1].filtered_measured_angle + self.InnerTransversalMotor_BR_sim2real_offset) * np.pi/180,
                            (self._legs_FL[1].filtered_measured_angle + self.InnerTransversalMotor_FL_sim2real_offset) * np.pi/180,
                            (self._legs_FL[2].filtered_measured_angle + self.OuterTransversalMotor_FL_sim2real_offset) * np.pi/180,
                            (self._legs_FR[1].filtered_measured_angle + self.InnerTransversalMotor_FR_sim2real_offset) * np.pi/180,
                            (self._legs_FR[2].filtered_measured_angle + self.OuterTransversalMotor_FR_sim2real_offset) * np.pi/180,
                            self._legs_BL[0].filtered_velocity * np.pi/180,
                            self._legs_BR[0].filtered_velocity * np.pi/180,
                            self._legs_FL[0].filtered_velocity * np.pi/180,
                            self._legs_FR[0].filtered_velocity * np.pi/180,
                            self._legs_BL[2].filtered_velocity * np.pi/180,
                            self._legs_BL[1].filtered_velocity * np.pi/180,
                            self._legs_BR[2].filtered_velocity * np.pi/180,
                            self._legs_BR[1].filtered_velocity * np.pi/180,
                            self._legs_FL[1].filtered_velocity * np.pi/180,
                            self._legs_FL[2].filtered_velocity * np.pi/180,
                            self._legs_FR[1].filtered_velocity * np.pi/180,
                            self._legs_FR[2].filtered_velocity * np.pi/180,
                            *orient_err,
                            *self.base_ang_vel
            ]).astype(np.float32)

            self.output = self.ort_sess.run(None, {'input': obs.astype(np.float32)})[0]

            if self._log_data:
                with open(self._data_save_path, 'a') as file:
                    file.write(','.join(map(str, [*obs, self.pole_ref, self._base_rotation, *[leg._measurement_buffer[-1] for leg in self._legs], start_time - self.policy_start_time, *self.output])) + '\n')

            if not self._use_laterals:
                self.output = [0,0,0,0,*self.output]
            
            interpol_angle = self._base_pos_filtered
            while  interpol_angle > np.pi:
                interpol_angle -= 2*np.pi
            while interpol_angle < -np.pi:
                interpol_angle += 2*np.pi

            orr_err = np.abs(quat_diff_rad(self._base_rotation, orient_ref))
            interpol_coeff = np.exp(-orr_err**2 / 0.002) 
                
            #interpol_coeff = np.exp(- (interpol_angle - self.pole_ref)**2/0.002)

            mapped_targets_BL = self.map_policy_targets(
                [self.output[0], self.output[5], self.output[4]],
                self._legs_BL,
                [self.LateralMotor_BL_sim2real_offset, self.InnerTransversalMotor_BL_sim2real_offset, self.OuterTransversalMotor_BL_sim2real_offset],
                interpol_coeff
            )
            
            mapped_targets_FL = self.map_policy_targets(
                [self.output[2], self.output[8], self.output[9]],
                self._legs_FL,
                [self.LateralMotor_FL_sim2real_offset, self.InnerTransversalMotor_FL_sim2real_offset, self.OuterTransversalMotor_FL_sim2real_offset],
                interpol_coeff
            )

            mapped_targets_BR = self.map_policy_targets(
                [self.output[1], self.output[7], self.output[6]],
                self._legs_BR,
                [self.LateralMotor_BR_sim2real_offset, self.InnerTransversalMotor_BR_sim2real_offset, self.OuterTransversalMotor_BR_sim2real_offset],
                interpol_coeff
            )

            mapped_targets_FR = self.map_policy_targets(
                [self.output[3], self.output[10], self.output[11]],
                self._legs_FR,
                [self.LateralMotor_FR_sim2real_offset, self.InnerTransversalMotor_FR_sim2real_offset, self.OuterTransversalMotor_FR_sim2real_offset],
                interpol_coeff
            )

            if not self._use_laterals:
                mapped_targets_BL[0] = 0.0 - self.LateralMotor_BL_sim2real_offset
                mapped_targets_FL[0] = 0.0 - self.LateralMotor_FL_sim2real_offset
                mapped_targets_BR[0] = 0.0 - self.LateralMotor_BR_sim2real_offset
                mapped_targets_FR[0] = 0.0 - self.LateralMotor_FR_sim2real_offset

            mapped_targets = mapped_targets_BL + mapped_targets_FL + mapped_targets_BR + mapped_targets_FR

            #####################################################################
            ## Apply actions
            #####################################################################

            for i, leg in enumerate(self._legs):
                if self._use_guidance_module:
                    leg.low_pass_guidance_second_order_module(mapped_targets[i])
                else:
                    leg.target_angle = mapped_targets[i]
            
            self.apply_motor_angles()

            #####################################################################
            ## Diagnostics
            #####################################################################

            # print("Orient error: ", orient_err)
            # print("Base rotation: ", self._base_rotation)
            # print("Pitch reference: ", self.pole_ref * 180/np.pi)

            end_time = rospy.get_time()
            time_used = (end_time-start_time)
            if abs(time_used*self._ros_rate ) -1 > 0:
                rospy.logwarn("Too slow: %f", time_used)
            

            # Terminate program after specific policy uptime
            if end_time - self.policy_start_time >= self._max_time:
                self.timeout = True
                self.mocap_subscriber.unregister()
        
        self.counter += 1
    
    def reference_callback(self, msg):
        if self._pitch_configuration: self._potmeter_ref = ((msg.data / 255.0) * (280 * np.pi / 180) - (140 * np.pi / 180))
        else: self._potmeter_ref = - (msg.data / 255.0) * (280 * np.pi / 180) - (140 * np.pi / 180)

    def on_press(self, key):
        print("ONE")
        if key== "s":
            print("TWO")
            self._robot_initialised = True
            stop_listening()

    def apply_motor_angles(self):
        for j, i in enumerate(self._legs_to_output_idx):
            self._motor_angles[i] = self._legs[j].target_angle
        self._motor_channels.angle = self._motor_angles # this writes to the PCA9685

    def map_policy_targets(self, policy_targets, legs, sim2real, interpol_coeff) -> List[float]:
        mapped_motor_targets = [0.0]*3
        for i, leg in enumerate(legs):
            mapped_target = policy_targets[i]*(self._motor_sim_limits[i][1] - self._motor_sim_limits[i][0])/2 + (self._motor_sim_limits[i][1] + self._motor_sim_limits[i][0]) / 2
            mapped_target = (1-interpol_coeff) * mapped_target + interpol_coeff * (leg.filtered_measured_angle + sim2real[i])
            mapped_target = np.clip(mapped_target, *self._motor_sim_limits[i])
            mapped_motor_targets[i] = mapped_target 

        transversal_sum = (mapped_motor_targets[1] + mapped_motor_targets[2]) - self._max_transversal_motor_diff
        if (transversal_sum < 0):
                mapped_motor_targets[1] -= transversal_sum/2
                mapped_motor_targets[2] -= transversal_sum/2
        
        transversal_sum = (mapped_motor_targets[1] + mapped_motor_targets[2]) - self._max_transversal_motor_sum
        if (transversal_sum > 0):
                mapped_motor_targets[1] -= transversal_sum/2
                mapped_motor_targets[2] -= transversal_sum/2

        for i in range(len(mapped_motor_targets)):
            mapped_motor_targets[i] -= sim2real[i]
        
        return mapped_motor_targets

    def run(self):
        self.rate = rospy.Rate(self._ros_rate) 

        # Run startup with slower velocity
        for leg in self._legs:
            leg.set_guidance_params(max_velocity=30)

        # Startup loop
        print("### Initialising legs ...")
        while not rospy.is_shutdown():
            start_time = rospy.get_time()

            for leg in self._legs:
                leg.update_filtered_measurement() # reads new measurement and updates filtered angle
                leg.update_filtered_velocity() # uses the last two filtered angles to calculate velocity


            self._legs_BL[0].low_pass_guidance_second_order_module(15 - self.LateralMotor_BL_sim2real_offset)
            self._legs_BL[1].low_pass_guidance_second_order_module(45 - self.InnerTransversalMotor_BL_sim2real_offset)
            self._legs_BL[2].low_pass_guidance_second_order_module(45 - self.OuterTransversalMotor_BL_sim2real_offset)
            self._legs_FL[0].low_pass_guidance_second_order_module(15 - self.LateralMotor_FL_sim2real_offset)
            self._legs_FL[1].low_pass_guidance_second_order_module(45 - self.InnerTransversalMotor_FL_sim2real_offset)
            self._legs_FL[2].low_pass_guidance_second_order_module(45 - self.OuterTransversalMotor_FL_sim2real_offset)
            self._legs_BR[0].low_pass_guidance_second_order_module(15 - self.LateralMotor_BR_sim2real_offset)
            self._legs_BR[1].low_pass_guidance_second_order_module(45 - self.InnerTransversalMotor_BR_sim2real_offset)
            self._legs_BR[2].low_pass_guidance_second_order_module(45 - self.OuterTransversalMotor_BR_sim2real_offset)
            self._legs_FR[0].low_pass_guidance_second_order_module(15 - self.LateralMotor_FR_sim2real_offset)
            self._legs_FR[1].low_pass_guidance_second_order_module(45 - self.InnerTransversalMotor_FR_sim2real_offset)
            self._legs_FR[2].low_pass_guidance_second_order_module(45 - self.OuterTransversalMotor_FR_sim2real_offset)

            self.apply_motor_angles()

            self.counter += 1
            if self.counter > self._ros_rate*10:
                break
        
        print(f"### Waiting for start signal to run policy for duration of {self._max_time} seconds, please press s key to begin...")
        listen_keyboard(on_press=self.on_press)
        while not self._robot_initialised: pass
        
        # Set guidance velocity back to normal
        for leg in self._legs:
            leg.set_guidance_params(max_velocity=self._guidance_max_velocity)
        
        # Main loop
        while not rospy.is_shutdown():
            if self.timeout:
                time.sleep(1)
                Eurepus()._motor_channels.fraction = [None]*16
                time.sleep(3)
                rospy.signal_shutdown("Time is out mf")
            self.rate.sleep()
            
                
def ctrlchandler(signal, frame):
    Eurepus()._motor_channels.fraction = [None]*16
    sys.exit(0)


if __name__ == '__main__':
    try:
        node = Eurepus()
        signal.signal(signal.SIGINT, ctrlchandler)
        node.run()
    except rospy.ROSInterruptException:
        pass
