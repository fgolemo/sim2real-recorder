import time

import numpy as np
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

for i in tqdm(range(40)):
    line_buffer = [i]
    for info in infos:
        line_buffer.append(getattr(poppy_remote.m2, info))
    out.append(line_buffer)
    time.sleep(.01)

np.savez("sensor-reading-fast.npz", m2=np.array(out))

poppy_remote.rest_posture.start()
time.sleep(3)
poppy_remote.rest_posture.stop()
