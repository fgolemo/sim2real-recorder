import random
import time

import numpy as np
from pypot.robot import from_remote
from tqdm import tqdm

poppy_remote = from_remote('flogo3.local', 4242)
print (dir(poppy_remote))

poppy_remote.rest_posture.start()
time.sleep(3)
poppy_remote.rest_posture.stop()


for j in range(5):

    for m in poppy_remote.motors:
        m.goal_speed = random.randint(10,100)
        m.goal_position = random.randint(-90,90)

    time.sleep(5)


poppy_remote.rest_posture.start()
time.sleep(3)
poppy_remote.rest_posture.stop()
