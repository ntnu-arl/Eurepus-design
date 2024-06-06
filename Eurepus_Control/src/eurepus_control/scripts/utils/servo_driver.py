# SPDX-FileCopyrightText: 2016 Radomir Dopieralski for Adafruit Industries
# SPDX-FileCopyrightText: 2017 Scott Shawcroft for Adafruit Industries
#
# SPDX-License-Identifier: MIT

"""
`adafruit_pca9685`
====================================================

Driver for the PCA9685 PWM control IC. Its commonly used to control servos, leds and motors.

.. seealso:: The `Adafruit CircuitPython Motor library
    <https://github.com/adafruit/Adafruit_CircuitPython_Motor>`_ can be used to control the PWM
    outputs for specific uses instead of generic duty_cycle adjustments.

* Author(s): Scott Shawcroft

Implementation Notes
--------------------

**Hardware:**

* Adafruit `16-Channel 12-bit PWM/Servo Driver - I2C interface - PCA9685
  <https://www.adafruit.com/product/815>`_ (Product ID: 815)

**Software and Dependencies:**

* Adafruit CircuitPython firmware for the ESP8622 and M0-based boards:
  https://github.com/adafruit/circuitpython/releases
* Adafruit's Bus Device library: https://github.com/adafruit/Adafruit_CircuitPython_BusDevice
* Adafruit's Register library: https://github.com/adafruit/Adafruit_CircuitPython_Register
"""

__version__ = "3.4.14"
__repo__ = "https://github.com/adafruit/Adafruit_CircuitPython_PCA9685.git"

import time

from adafruit_register.i2c_struct import UnaryStruct, Struct
from adafruit_register.i2c_struct_array import StructArray
from adafruit_bus_device import i2c_device

try:
    from typing import Optional, Type, List
    from types import TracebackType
    from busio import I2C
except ImportError:
    pass


class PWMChannel:
    """A single PCA9685 channel that matches the :py:class:`~pwmio.PWMOut` API.

    :param PCA9685 pca: The PCA9685 object
    :param int index: The index of the channel
    """

    def __init__(self, pca: "PCA9685"):
        self._pca = pca
        self.set_pulse_width_range()
        self.set_actuation_range()

    @property
    def duty_cycle(self) -> List[int]:
        """16 bit value that dictates how much of one cycle is high (1) versus low (0). 0xffff will
        always be high, 0 will always be low and 0x7fff will be half high and then half low.
        """
        pwm = self._pca.pwm_regs
        duty_cycles = [None]*16
        for i in range(16):
            if pwm[2*i] == 0x1000:
                duty_cycles[i] = 0xFFFF
            elif pwm[2*i+1] == 0x1000:
                duty_cycles[i] = 0x0000
            else:
                duty_cycles[i] = pwm[2*i+1] << 4
        return duty_cycles

    @duty_cycle.setter
    def duty_cycle(self, value: List[int]) -> None:
        # value should be inside 0x0000 and 0xFFFF
        new_values = [None]*32
        for i in range(16):
            if value[i] == 0xFFFF:
                new_values[2*i] = 0x1000
                new_values[2*i+1] = 0
            elif value[i] < 0x0010:
                new_values[2*i] = 0
                new_values[2*i+1] = 0x1000
            else:
                new_values[2*i] = 0
                new_values[2*i+1] = value[i] >> 4
        
        self._pca.pwm_regs = new_values

    @property
    def fraction(self) -> Optional[float]:
        """Pulse width expressed as fraction between 0.0 (`min_pulse`) and 1.0 (`max_pulse`).
        For conventional servos, corresponds to the servo position as a fraction
        of the actuation range. Is None when servo is diabled (pulsewidth of 0ms).
        """

        duty_cycles = self.duty_cycle

        for i in range(16):
            if duty_cycles[i] == 0:
                duty_cycles[i] = None
            else:
                duty_cycles[i] = (duty_cycles[i] - self._min_duty) / self._duty_range
        
        return duty_cycles

    @fraction.setter
    def fraction(self, value: Optional[List[float]]) -> None:
        if value is None:
            self.duty_cycle = [0]*16  # disable the motor
            return
        if not 0.0 <= any(value) <= 1.0:
            raise ValueError("Must be 0.0 to 1.0")
        
        duty_cycles = [None]*16
        for i in range(16):
            if value[i] is None:
                duty_cycles[i] = 0
            else:
                duty_cycles[i] = self._min_duty + int(value[i] * self._duty_range)

        self.duty_cycle = duty_cycles

    @property
    def angle(self) -> Optional[List[float]]:
        """The servo angle in degrees. Must be in the range ``0`` to ``actuation_range``.
        Is None when servo is disabled."""
        angle = self.fraction
        for i in range(16):
            if angle[i] != None:
                angle[i] = angle[i] * self._actuation_range

        return angle

    @angle.setter
    def angle(self, new_angle: Optional[List[int]]) -> None:

        if new_angle is None:
            self.fraction = None
            return
        
        if not 0 <= any(new_angle) <= self._actuation_range:
            raise ValueError("Angle out of range")
        
        fractions = [None]*16
        for i in range(16):
            if new_angle[i] != None:
                fractions[i] = new_angle[i] / self._actuation_range
        self.fraction = fractions

    def set_pulse_width_range(
        self, min_pulse: int = 750, max_pulse: int = 2250
    ) -> None:
        """Change min and max pulse widths."""
        self._min_duty = int((min_pulse * self._pca.frequency) / 1000000 * 0xFFFF)
        max_duty = (max_pulse * self._pca.frequency) / 1000000 * 0xFFFF
        self._duty_range = int(max_duty - self._min_duty)

    def set_actuation_range(self, actuation_range: int = 180) -> None:
        """Change the actuation range."""
        self._actuation_range = actuation_range


class PCA9685:
    """
    Initialise the PCA9685 chip at ``address`` on ``i2c_bus``.

    The internal reference clock is 25mhz but may vary slightly with environmental conditions and
    manufacturing variances. Providing a more precise ``reference_clock_speed`` can improve the
    accuracy of the frequency and duty_cycle computations. See the ``calibration.py`` example for
    how to derive this value by measuring the resulting pulse widths.

    :param ~busio.I2C i2c_bus: The I2C bus which the PCA9685 is connected to.
    :param int address: The I2C address of the PCA9685.
    :param int reference_clock_speed: The frequency of the internal reference clock in Hertz.
    """

    # Registers:
    mode1_reg = UnaryStruct(0x00, "<B")
    mode2_reg = UnaryStruct(0x01, "<B")
    prescale_reg = UnaryStruct(0xFE, "<B")
    pwm_regs = Struct(0x06, "<HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")

    def __init__(
        self,
        i2c_bus: I2C,
        *,
        address: int = 0x40,
        reference_clock_speed: int = 25000000,
    ) -> None:
        self.i2c_device = i2c_device.I2CDevice(i2c_bus, address)
        self.reference_clock_speed = reference_clock_speed
        """The reference clock speed in Hz."""
        self.channels = PWMChannel(self)
        """Sequence of 16 `PWMChannel` objects. One for each channel."""
        
        self.reset()

    def reset(self) -> None:
        """Reset the chip."""
        self.mode1_reg = 0x00  # Mode1

    @property
    def frequency(self) -> float:
        """The overall PWM frequency in Hertz."""
        prescale_result = self.prescale_reg
        if prescale_result < 3:
            raise ValueError(
                "The device pre_scale register (0xFE) was not read or returned a value < 3"
            )
        return self.reference_clock_speed / 4096 / (prescale_result + 1)

    @frequency.setter
    def frequency(self, freq: float) -> None:
        prescale = int(self.reference_clock_speed / 4096.0 / freq + 0.5) - 1
        if prescale < 3:
            raise ValueError("PCA9685 cannot output at the given frequency")
        old_mode = self.mode1_reg  # Mode 1
        self.mode1_reg = (old_mode & 0x7F) | 0x10  # Mode 1, sleep
        self.prescale_reg = prescale  # Prescale
        self.mode1_reg = old_mode  # Mode 1
        time.sleep(0.005)
        # Mode 1, autoincrement on, fix to stop pca9685 from accepting commands at all addresses
        self.mode1_reg = old_mode | 0xA0

    def __enter__(self) -> "PCA9685":
        return self

    def __exit__(
        self,
        exception_type: Optional[Type[type]],
        exception_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.deinit()

    def deinit(self) -> None:
        """Stop using the pca9685."""
        self.reset()
