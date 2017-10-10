import time
from pypot.creatures import PoppyErgoJr
import numpy as np
from tqdm import tqdm

poppy = PoppyErgoJr(simulator='vrep')

# from pypot.vrep import from_vrep
#
# config = '/usr/local/lib/python2.7/dist-packages/poppy_ergo_jr/configuration/poppy_ergo_jr.json'
# scene = '/usr/local/lib/python2.7/dist-packages/poppy_ergo_jr/vrep-scene/poppy_ergo_jr.ttt'
# # scene = '/home/florian/REMOTE_API_TEMPFILE_5525.ttt'
#
# collision_objects = ["base_link_visual", "lamp_visual"]
#
# # poppy = from_vrep(config, '127.0.0.1', 19997, scene, tracked_collisions=collision_objects)
# poppy = from_vrep(config, '127.0.0.1', 19997, scene)


# https://github.com/poppy-project/poppy-torso/blob/ff6254355ce18a26f58654f5abc82485a7a22d13/software/doc/tutorial/Poppy%20Torso%20interacting%20with%20objects%20in%20V-REP%20using%20Pypot.ipynb

print (dir(poppy))

poppy.reset_simulation()
time.sleep(1)
poppy.rest_posture.start()
time.sleep(1)
poppy.start_sync()
# poppy.stop_simulation()

# pos = {"m1": 0, "m2": 45, "m3": 45, "m4": 0, "m5": 45, "m6": 45}
# poppy.goto_position(pos, 5., wait=False)
#
# for i in range(500):
#     print (dir(poppy.motors[1]))
#     quit()

infos = ['present_load', 'present_position', 'present_speed', 'present_temperature', 'present_voltage']
for info in infos:
    print (info, getattr(poppy.m2, info))

poppy.m2.goal_speed = 40
poppy.m2.goal_position = +20

out = []

for i in tqdm(range(1000)):
    line_buffer = [i]
    for info in infos:
        line_buffer.append(getattr(poppy.m2, info))
    out.append(line_buffer)
    time.sleep(.01)

np.savez("sensor-reading.npz", m2=np.array(out))


