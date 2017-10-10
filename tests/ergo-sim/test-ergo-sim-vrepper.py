import os
import time

import numpy as np

try:
    from vrepper.vrepper import vrepper
except ImportError as e:
    raise os.error.DependencyNotInstalled(
        "{}. (HINT: you can install VRepper dependencies with 'pip install vrepper.)'".format(e))

import logging

logger = logging.getLogger(__name__)

JOINT_LIMITS = [
    (-150, 150),
    (-90, 125),
    (-90, 90),
    (-90, 90),
    (-90, 90),
    (-90, 90)
]

JOINT_LIMITS_MAXMIN = [-150, 150]

FORCE_LIMITS = [-5.0, 5.0]

REST_POS = [0, -90, 35, 0, 55, 0]

minima = [JOINT_LIMITS[i][0] for i in range(6)]
maxima = [JOINT_LIMITS[i][1] for i in range(6)]


class Experiment():
    def _startEnv(self, headless, scene=None):
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

    def _restPos(self):
        self.venv.stop_simulation()
        self.venv.start_simulation(is_sync=False)

        for i, m in enumerate(self.motors):
            m.set_position_target(REST_POS[i])
        time.sleep(.5)

        self.venv.make_simulation_synchronous(True)

    def _self_observe(self):
        pos = []
        forces = []
        for m in self.motors:
            pos.append(m.get_joint_angle())
            forces += m.get_joint_force()

        tip_pos = self.tip.get_position()

        observation = np.array(pos + forces + tip_pos).astype('float32')
        return observation

    def _gotoPos(self, pos, vel):
        for i, m in enumerate(self.motors):
            m.set_position_target(pos[i])
            if vel is not None:
                m.set_force(vel)

    @staticmethod
    def _clipActions(actions):
        a = []
        for i, action in enumerate(actions):
            a.append(np.clip(action, minima[i], maxima[i]))
        return np.array(a)

    def _step(self, actions, velocity=None):
        actions = self._clipActions(actions)

        # step
        self._gotoPos(actions, velocity)
        self.venv.step_blocking_simulation()

        # observe again
        observation = self._self_observe()

        return observation

    def _close(self):
        self.venv.stop_simulation()
        self.venv.end()

    @staticmethod
    def randomAction(step=1, oldActions=None):
        out = np.random.uniform(-step, step, 6)
        if oldActions is None:
            return out
        return out + oldActions


if __name__ == '__main__':
    actions = np.zeros(6)

    exp = Experiment()

    exp._startEnv(False, '/usr/local/lib/python2.7/dist-packages/poppy_ergo_jr/vrep-scene/poppy_ergo_jr.ttt')

    raw_input("change simulation step size to 10ms and press ENTER...")

    exp._restPos()

    out = []

    # for i in range(100):
    #     if i % 10 == 0:
    #         actions = exp.randomAction(step=30, oldActions=actions)
    #     exp._step(actions)

    for i in range(0, 50):
        obs = exp._step([0, 30, 35, 0, 55, 0], .1)
        print (np.around(obs[-3:], 5))
        out.append(obs)
        print (i)

    print ("=============")

    for i in range(50, 100):
        obs = exp._step(REST_POS)
        out.append(obs)
        print(i)

    print ("=============")

    for i in range(100, 150):
        # obs = exp._step([0, 90, 90, 0, 55, 0], 1000)
        obs = exp._step([0, 30, 35, 0, 55, 0], 1000)
        out.append(obs)
        print (np.around(obs[-3:], 5))
        print(i)

        # np.savez("sensor-reading.npz", sensors=np.array(out))
