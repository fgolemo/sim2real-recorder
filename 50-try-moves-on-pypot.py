import time

from s2rr.movements.dataset import Dataset
from poppy.creatures import PoppyErgoJr

from recorder.experiment import Experiment

DATASET_PATH_CLEAN = "data/recording1_clean.npz"
EPISODE = 0

# RESULT: MOTOR 0 IS INVERTED
# RESULT: MOTOR 3 IS INVERTED


exp = Experiment()
exp.startEnv(headless=False)

ds = Dataset()
ds.load(DATASET_PATH_CLEAN)

poppy = PoppyErgoJr(simulator='vrep')

poppy.rest_posture.start()
time.sleep(2)
poppy.rest_posture.stop()
time.sleep(.1)

pos = ds.moves[EPISODE,0,0,:]
print (pos)
for motor_idx, motor in enumerate(poppy.motors):
    motor.goal_position = pos[motor_idx]

pos[0] *= -1
pos[3] *= -1
for _ in range(15):
    exp.step(pos)

# =============

pos = ds.moves[EPISODE,1,0,:]
print (pos)
for motor_idx, motor in enumerate(poppy.motors):
    motor.goal_position = pos[motor_idx]

pos[0] *= -1
pos[3] *= -1
for _ in range(15):
    exp.step(pos)

# =============

pos = ds.moves[EPISODE,2,0,:]
print (pos)
for motor_idx, motor in enumerate(poppy.motors):
    motor.goal_position = pos[motor_idx]

pos[0] *= -1
pos[3] *= -1
for _ in range(15):
    exp.step(pos)







