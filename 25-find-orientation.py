from movements.constants import REST_POS
from movements.dataset import Dataset
from recorder.experiment import Experiment
from tqdm import tqdm
import numpy as np


exp = Experiment()

exp.startEnv(headless=False)
obs = exp.self_observe()
print ("resting", obs[-3:])
#
#
# pos = [0,0,0,0,0,0]
# for i in range(10):
#     obs = exp.step(pos)
# print ("normal, front", obs[-3:])
#
# pos = [90,0,0,0,0,0]
# for i in range(10):
#     obs = exp.step(pos)
# print ("right", obs[-3:])
#
# pos = [-90,0,0,0,0,0]
# for i in range(10):
#     obs = exp.step(pos)
# print ("left", obs[-3:])
#
# pos = [-90,0,0,90,0,0] # WAT? One of the motors is inverted!
# for i in range(10):
#     obs = exp.step(pos)
# print ("back", obs[-3:])


exp.close()

# ('resting', array([ 0.00148483,  0.01526723,  0.12403183], dtype=float32))
# ('front', array([ 0.0034607 , -0.10566922,  0.15233842], dtype=float32))
# ('right', array([-0.13970707,  0.02941428,  0.14521824], dtype=float32))
# ('left', array([ 0.14582232,  0.0299342 ,  0.14177077], dtype=float32))
# ('back', array([ 0.00358735,  0.16868548,  0.1508483 ], dtype=float32))