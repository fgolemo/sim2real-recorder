import os
import numpy as np
import time

from vrepper.core import vrepper


class Simulator(object):

    def __init__(self):
        self.venv = vrepper(headless=True)
        self.venv.start()
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.venv.load_scene(current_dir + '/../scenes/poppy_ergo_jr.ttt')
        self.motors = []
        for i in range(6):
            motor = self.venv.get_object_by_name('m{}'.format(i + 1), is_joint=True)
            self.motors.append(motor)

        self.cam = self.venv.get_object_by_name("cam")

    def set_motors(self, positions, speeds=None):
        for i, m in enumerate(self.motors):
            target = positions[i]
            if i == 0:
                target *= -1
            m.set_position_target(target)
            if speeds is not None:
                m.set_velocity(speeds[i])

    def get_motors(self):
        out_pos = np.zeros(6, dtype=np.float32)
        out_vel = np.zeros(6, dtype=np.float32)
        for i, m in enumerate(self.motors):
            angle = m.get_joint_angle()
            if i == 0:
                angle *= -1
            out_pos[i] = angle
            out_vel[i] = m.get_joint_velocity()[0]

        return out_pos, out_vel

    def reset_sim(self, initial_position=None):
        assert len(initial_position) == 6

        self.venv.stop_simulation()
        self.venv.start_simulation(is_sync=False)  # start in realtime mode to set initial position

        if initial_position is not None:
            # init position for this turn
            self.set_motors(initial_position)
        time.sleep(.2)
        self.venv.make_simulation_synchronous(True)

    def close(self):
        self.venv.stop_simulation()
        time.sleep(0.1)
        self.venv.end()

    def step(self, pos, vel):
        self.set_motors(pos, vel)
        self.venv.step_blocking_simulation()
