import time
from poppy_ergo_jr import PoppyErgoJr
import numpy as np

poppy = PoppyErgoJr()

out_buffer = np.zeros((6,3), dtype=np.float32)

def getRobotData():
    for motor_idx, motor in enumerate(poppy.motors):
        out_buffer[motor_idx] = [
            motor.present_position,
            motor.present_speed,
            motor.present_load
            # motor.present_voltage
            # motor.present_temperature
        ]
    return out_buffer

SECONDS_OF_RECORDING = 3.0

for i in range(10):
    frames = []
    time_start = time.time()
    while True:
        robot_data = getRobotData()
        frames.append(robot_data)
        # print (robot_data)
        time.sleep(0.01) # this caps FPS at aorund 94 - so we should reserve 100 elements in memory
        time_current = time.time() - time_start
        if time_current >= SECONDS_OF_RECORDING:
            break
    print("{} frames / {} fps".format(len(frames), round(len(frames) / SECONDS_OF_RECORDING, 2)))




