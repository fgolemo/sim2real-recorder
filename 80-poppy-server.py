import time

import numpy as np
from pypot.creatures import PoppyErgoJr

from config.constants import *
from exchange.server import Server

poppy = PoppyErgoJr()

server = Server()

server.welcome()

line_buffer = np.zeros((6, 3), dtype=np.float32)
episode_buffer = np.zeros((SECONDS_OF_RECORDING * MAX_ROBO_FPS, NUMBER_OF_ACTIONS_PER_EPISODE, 6, 3))


def getRobotData():
    for motor_idx, motor in enumerate(poppy.motors):
        line_buffer[motor_idx] = [
            motor.present_position,
            motor.present_speed,
            motor.present_load
        ]
    return line_buffer


def msg_to_actions(msg):
    actions = msg.split("|")
    assert len(actions) == NUMBER_OF_ACTIONS_PER_EPISODE
    actions_out = []
    for a in actions:
        actions_out.append(a.split(";"))
    return actions_out


def run_action_on_robot(action):
    for motor_idx, motor in enumerate(poppy.motors):
        motor.goal_position = action[motor_idx]


def robot_rest():
    poppy.rest_posture.start()
    time.sleep(ROBO_REST_TIMER)
    poppy.rest_posture.stop()


def handle_message(msg, send):
    # whenever the server gets a string msg:

    # split string into 3 actions
    actions = msg_to_actions(msg)
    print ("got actions:")
    for a in actions:
        print (a)

    # run 3 actions, while recording into buffer
    for action_idx, action in enumerate(actions):
        frames = []
        time_start = time.time()
        action_was_run = False
        while True:
            robot_data = getRobotData()
            frames.append(robot_data)
            time.sleep(0.01)  # this caps FPS at around 94 - so we should reserve 100 elements in memory
            if not action_was_run:
                run_action_on_robot(action)
                action_was_run = True

            time_current = time.time() - time_start
            if time_current >= SECONDS_OF_RECORDING:
                break
        # print("{} frames / {} fps".format(len(frames), round(len(frames) / SECONDS_OF_RECORDING, 2)))
        frames = np.array(frames)
        print (frames.shape)
        print (episode_buffer[:len(frames), action_idx, :, :].shape)
        episode_buffer[:len(frames), action_idx, :, :] = frames

    # go to resting position
    robot_rest()

    # send data
    send(episode_buffer)

    episode_buffer.fill(0.0)


server.register_callback(handle_message)
server.start_main_loop()
