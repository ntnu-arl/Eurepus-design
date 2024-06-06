#!/usr/bin/env python3

import numpy as np

def quat_from_euler_xyz(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    quat = np.array([qw, qx, qy, qz])

    # if qy < 0:
    #     return - quat
    # else:
    #     return quat
    return quat

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mul(a, b):
    assert a.shape == b.shape

    w1, x1, y1, z1 = a[0], a[1], a[2], a[3]
    w2, x2, y2, z2 = b[0], b[1], b[2], b[3]

    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))

    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = np.array([w, x, y, z])

    return quat

def get_euler_xyz(q):
    qw, qx, qy, qz = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[qw] * q[qx] + q[qy] * q[qz])
    cosr_cosp = q[qw] * q[qw] - q[qx] * q[qx] - q[qy] * q[qy] + q[qz] * q[qz]
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[qw] * q[qy] - q[qz] * q[qx])
    if np.abs(sinp) >= 1:
        pitch = np.sign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)


    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[qw] * q[qz] + q[qx] * q[qy])
    cosy_cosp = q[qw] * q[qw] + q[qx] * q[qx] - q[qy] * q[qy] - q[qz] * q[qz]
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)


def quat_diff_rad(a, b):
    """
    Get the difference in radians between two quaternions.
    """

    b_conj = quat_conjugate(b)
    mul = quat_mul(a, b_conj)
    # 2 * torch.acos(torch.abs(mul[:, -1]))
    return 2.0 * np.arcsin(np.linalg.norm(mul[1:]))

def ang_vel_from_quat(q1, q2, dt):
    q_dot = (q2-q1)/dt
    omega = 2 * quat_mul(q_dot, quat_conjugate(q2))[1:] 
    return omega

def trapezoidal_velocity(current_pos, start_pos, end_pos, max_speed, f1, f2):

    range_length = end_pos-start_pos

    ramp_up_range = [start_pos, start_pos + range_length*f1]
    const_speed_range = [start_pos + range_length*f1 , end_pos - range_length*f2]
    ramp_down_range = [end_pos - range_length*f2, end_pos]

    # Check if the current position is within the ramp-up range
    if is_between(current_pos, ramp_up_range):

        # Linear interpolation for ramp-up
        speed = (current_pos - ramp_up_range[0]) / (ramp_up_range[1] - ramp_up_range[0]) * max_speed

    # Check if the current position is within the constant speed range
    elif is_between(current_pos, const_speed_range):
        # Constant speed within the range
        speed = max_speed

    # Check if the current position is within the ramp-down range
    elif is_between(current_pos, ramp_down_range):
        # Linear interpolation for ramp-down
        speed = (ramp_down_range[1] - current_pos) / (ramp_down_range[1] - ramp_down_range[0]) * max_speed

    else:
        # Outside of the defined ranges, speed is 0
        speed = 0.01

    return speed

def is_between(number, range):
    return np.logical_and(np.minimum(range[0], range[1]) <= number, number <= np.maximum(range[0], range[1]))
