#!/usr/bin/env python3

import rospy
import board 
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import adafruit_tca9548a
import adafruit_pca9685
from adafruit_servokit import ServoKit

from std_msgs.msg import Float64


import Leg

class ServoMotor:
    def __init__(self):
        rospy.init_node('servo_verification')

        # Create the I2C bus
        self._i2c = busio.I2C(board.GPIOX_18, board.GPIOX_17, frequency=1000000)

        # Create the Multiplexer object
        self._tca = adafruit_tca9548a.TCA9548A(self._i2c)

        # Create the ADC objects
        self._ads_FL = ADS.ADS1115(self._tca[3])
        self._ads_FL_channels = [AnalogIn(self._ads_FL, ADS.P3)]

        # Create the ServoKit object
        self._kit = ServoKit(channels=16, i2c=self._tca[7], frequency=333)

        # Set the frequency of the PWM signal
        self._kit.servo[0].set_pulse_width_range(500, 2500)
        self._kit.servo[0].actuation_range = 180

        self._last_angle = 0

        self._mongo_counter = 0

        rospy.loginfo("Servo Motor node initialised")

    def run(self):
        rate = rospy.Rate(120)  # 10 Hz, adjust as needed

        voltage_pub = rospy.Publisher('meas/analogue_feedback', Float64, queue_size=10)

        while not rospy.is_shutdown():

            voltage = self._ads_FL_channels[0].voltage
            rospy.loginfo("Voltage: %f", voltage)

            voltage_pub.publish(voltage)

            self._mongo_counter += 1
            if (self._mongo_counter % 20 == 0):
                self._mongo_counter = 0

                if self._last_angle == 0:
                    # self._kit.servo[0].angle = 180
                    self._last_angle = 180
                
                else:
                    # self._kit.servo[0].angle = 0
                    self._last_angle = 0
            

            rate.sleep()                                                         

if __name__ == '__main__':
    try:
        node = ServoMotor()
        node.run()
    except rospy.ROSInterruptException:
        pass
