from movements.constants import REST_POS
from movements.dataset import Dataset
from recorder.experiment import Experiment
from tqdm import tqdm
import numpy as np


exp = Experiment()

exp.startEnv(headless=False)

obs = exp.self_observe()

while True:
    print ("Enter 6 motor commands separated by comma (like '0,0,15,9,-10,5') or type in 'quit'" )
    command = raw_input(">")
    if command == "quit":
        break
    command_split = command.split(",")
    if len(command_split) != 6:
        print ("That doesn't look like 6 values or 'quit'.")
        continue
    pos = [float(c) for c in command_split]
    for i in range(20):
        obs = exp.step(pos)
    print ("Position: ", obs[-3:])

exp.close()



# right middle
# >110,30,30,-10,20,20
# ('Position: ', array([-0.09069847,  0.10101081,  0.05190547], dtype=float32))
#
# left middle
# >-120,20,30,25,10,20
# ('Position: ', array([ 0.07237665,  0.13728595,  0.05413443], dtype=float32))
#
# min height
# >-120,0,0,70,30,30
# ('Position: ', array([-0.01698578,  0.1418708 ,  0.0962072 ], dtype=float32))
#
# back
# >-120,0,0,70,25,25
# ('Position: ', array([-0.01811513,  0.14891243,  0.1032753 ], dtype=float32))
#
# front
# >-120,0,0,70,90,90
# ('Position: ', array([-0.00050217,  0.04738161,  0.10128222], dtype=float32))
#
# min height 2
# >0,30,30,0,20,20
# ('Position: ', array([ -8.12401995e-05,  -8.38212520e-02,   5.08071445e-02], dtype=float32))
