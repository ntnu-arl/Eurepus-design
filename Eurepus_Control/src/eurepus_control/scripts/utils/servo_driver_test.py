#!/usr/bin/env python3

import rospy
import board 
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import adafruit_tca9548a
from servo_driver import PCA9685
from adafruit_ads1x15.ads1x15 import Mode

from std_msgs.msg import Float64


import Leg

class ServoMotor:
    def __init__(self):
        rospy.init_node('servo_driver_test')

        # Create the I2C bus
        self._i2c = busio.I2C(board.GPIOX_18, board.GPIOX_17, frequency=1000000)

        # Create the Multiplexer object
        self._tca = adafruit_tca9548a.TCA9548A(self._i2c)

        # Create the pca object
        self._pca = PCA9685(self._tca[7], reference_clock_speed=26413670)
        self._pca.frequency = 333
    
        self._motor_channels = self._pca.channels
        self._motor_channels.set_pulse_width_range(500, 2500)
        self._motor_channels.set_actuation_range(173.5)

        # Create ADC for timing testing
        self._ads_FL = ADS.ADS1115(self._tca[3])
        self._ads_FL.mode = Mode.CONTINUOUS

        self._ads_FL.data_rate = 860
        self._ads_FL_channels = [AnalogIn(self._ads_FL, ADS.P0)]

        rospy.loginfo("Servo Motor node initialised")

        self._angles = [60]*16

    def run(self):
        rate = rospy.Rate(120)  # 10 Hz, adjust as needed

        while not rospy.is_shutdown():
            for i in range(12):
                voltage = self._ads_FL_channels[0].voltage


            self._motor_channels.angle = self._angles
            
        
            rate.sleep()                                                         

if __name__ == '__main__':
    try:
        node = ServoMotor()
        node.run()
    except rospy.ROSInterruptException:
        pass
