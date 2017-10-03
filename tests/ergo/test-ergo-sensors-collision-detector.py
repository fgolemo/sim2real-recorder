import time

import numpy as np
import math
from pypot.robot import from_remote
from tqdm import tqdm

poppy_remote = from_remote('flogo3.local', 4242)
print (dir(poppy_remote))

poppy_remote.rest_posture.start()
time.sleep(3)
print (dir(poppy_remote.m1))

infos = ['present_load', 'present_position', 'present_speed', 'present_temperature', 'present_voltage']
for info in infos:
    print (info, getattr(poppy_remote.m2, info))

poppy_remote.m2.goal_speed = 40
poppy_remote.m2.goal_position = +20

out = []

motor_history_2 = []

COLLISION_POS_THRESHOLD = .2
COLLISION_POS_DELAY = 50


def has_collision(motor):
    current_pos = motor.present_position
    motor_history_2.append(current_pos)

    if len(motor_history_2) < COLLISION_POS_DELAY:
        return False

    diff1 = abs(motor_history_2[-4] - motor_history_2[-3])
    diff2 = abs(motor_history_2[-3] - motor_history_2[-2])
    diff3 = abs(motor_history_2[-2] - motor_history_2[-1])

    diff_avg = (diff1 + diff2  + diff3) / 3
    print (diff_avg)

    if diff_avg < COLLISION_POS_THRESHOLD:
        return True

    return False


for i in tqdm(range(1000)):
    if has_collision(poppy_remote.m2):
        print (i)
        break
    time.sleep(.01)


poppy_remote.rest_posture.start()
time.sleep(3)
poppy_remote.rest_posture.stop()
