import time

import numpy as np
from pypot.robot import from_remote
from tqdm import tqdm

poppy_remote = from_remote('flogo3.local', 4242)
print (dir(poppy_remote))

poppy_remote.rest_posture.start()
time.sleep(3)
poppy_remote.rest_posture.stop()


target_pos = [60, 20, -30, -60, -45, 55]

for i, m in enumerate(poppy_remote.motors):
    m.goal_speed = 40
    m.goal_position = target_pos[i]

time.sleep(5)

target_pos = [-60, -20, 30, 60, 45, -55]

for i, m in enumerate(poppy_remote.motors):
    m.goal_speed = 40
    m.goal_position = target_pos[i]

time.sleep(5)


poppy_remote.rest_posture.start()
time.sleep(3)
poppy_remote.rest_posture.stop()
