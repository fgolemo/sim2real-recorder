import time

import numpy as np
from pypot.robot import from_remote

from recorder.params import ROBOT_PORT, ROBOT_HOST


class Robot():
    def __init__(self, initial_speed=100):
        self.poppy = from_remote(ROBOT_HOST, ROBOT_PORT)

        for motor_idx, motor in enumerate(self.poppy.motors):
            motor.goal_speed = initial_speed

        self.data_buffer = np.zeros((6, 6), np.float32)
        self.rest()

    def rest(self):
        self.poppy.rest_posture.start()
        time.sleep(3)
        self.poppy.rest_posture.stop()

    def close(self):
        self.rest()

    def get_data(self):
        for motor_idx, motor in enumerate(self.poppy.motors):
            self.data_buffer[motor_idx] = [
                motor.present_load,
                motor.present_position,
                motor.present_speed,
                motor.present_temperature,
                motor.present_voltage,
                motor.moving_speed
            ]
        return self.data_buffer

    def move(self, joints, is_inverted=True, speeds=[]):
        joints = [float(j) for j in joints]
        if is_inverted:
            joints[0] *= -1
            joints[3] *= -1

        joints_rounded = np.around(joints, 2)
        for motor_idx, motor in enumerate(self.poppy.motors):
            # motor.goal_speed = random.randint(10, 100)
            motor.goal_position = joints_rounded[motor_idx]

        time.sleep(5)
