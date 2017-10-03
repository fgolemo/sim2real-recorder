import os
import time

import numpy as np

from movements.constants import REST_POS, JOINT_LIMITS

try:
    from vrepper.vrepper import vrepper
except ImportError as e:
    raise os.error.DependencyNotInstalled(
        "{}. (HINT: you can install VRepper dependencies with 'pip install vrepper.)'".format(e))



class Experiment():
    def startEnv(self, headless, scene=None):
        self.venv = vrepper(headless=headless)
        self.venv.start()
        current_dir = os.path.dirname(os.path.realpath(__file__))
        if scene is None:
            scene = current_dir + '/../scenes/poppy_ergo_jr.ttt'

        self.venv.load_scene(scene)

        self.tip = self.venv.get_object_by_name('lamp_visual', is_joint=False)

        self.motors = []
        for i in range(6):
            motor = self.venv.get_object_by_name('m{}'.format(i + 1), is_joint=True)
            self.motors.append(motor)
        self.restPos()

    def restPos(self):
        self.venv.stop_simulation()
        self.venv.start_simulation(is_sync=False)

        for i, m in enumerate(self.motors):
            m.set_position_target(REST_POS[i])
        time.sleep(.5)

        self.venv.make_simulation_synchronous(True)

    def self_observe(self):
        pos = []
        forces = []
        for m in self.motors:
            pos.append(m.get_joint_angle())
            forces += m.get_joint_force()

        tip_pos = self.tip.get_position()

        observation = np.array(pos + forces + tip_pos).astype('float32')
        return observation

    def gotoPos(self, pos):
        for i, m in enumerate(self.motors):
            m.set_position_target(pos[i])

    @staticmethod
    def clipActions(actions):
        a = []
        for i, action in enumerate(actions):
            a.append(np.clip(action, JOINT_LIMITS[i][0], JOINT_LIMITS[i][1]))
        return np.array(a)

    def step(self, actions):
        actions = self.clipActions(actions)

        # step
        self.gotoPos(actions)
        self.venv.step_blocking_simulation()

        # observe again
        observation = self.self_observe()

        return observation

    def close(self):
        self.venv.stop_simulation()
        self.venv.end()

    @staticmethod
    def randomAction(step=1, oldActions=None):
        out = np.random.uniform(-step, step, 6)
        if oldActions is None:
            return out
        return out + oldActions

