import numpy as np

from s2rr.config.constants import WRITE_EVERY_N_EPISODES

from s2rr.movements.dataset import Dataset


FILE_NAME = "/lindata/datasets/sim2real/data_dump_{}_aligned.npz"
MOVES = "data/recording1_clean.npz"

# MOTOR 0 & 3 ARE INVERTED

ds = Dataset()
ds.load(MOVES)

# good: 0 1 2 3 4 5 6 7

for FILE_IDX in range(148):
    # FILE_IDX = 35
    data = np.load(FILE_NAME.format(FILE_IDX), encoding="bytes")
    real_positions, real_speeds = data["real_positions"], data["real_speeds"]



    for OFFSET in range(-20, 21):
        diffs = []
        for RECORDING in range(10):


            # OFFSET = -9 # IDK why

            current_episode = FILE_IDX*WRITE_EVERY_N_EPISODES+RECORDING+OFFSET

            cmds = ds.getUniqueActionsForEpisode(current_episode)
            cmds = cmds[:,0,:]

            # print("cmds.shape",cmds.shape)
            # print ("real_pos",real_positions.shape)
            # print ("real_speeds",real_speeds.shape)

            episode_pos = real_positions[RECORDING]
            episode_spd = real_speeds[RECORDING]

            # print ("episode_pos", episode_pos.shape)
            # print ("episode_spd", episode_spd.shape)

            old_speed = episode_spd[0]
            final_positions = []
            for step in range(len(episode_spd)-1):
                if episode_spd[step+1] != old_speed:
                    # new action
                    old_speed = episode_spd[step + 1]
                    final_positions.append(episode_pos[step,:,0])
            final_positions.append(episode_pos[-1,:,0])
            final_positions = np.array(final_positions)
            # print ("final_positions.shape",final_positions.shape)

            for speed in range(3):
                for joint in range(6):
                    # print ("{} - action\joint: {}\t{}".format(
                    #     speed,
                    #     np.around(cmds[speed, joint], 2),
                    #     np.around(final_positions[speed, joint], 2)
                    # ))
                    diffs.append(cmds[speed, joint] - final_positions[speed, joint])

        diff = np.mean(np.power(diffs,2))
        if diff < 1000:
            print ("file {}, offset {}".format(
                FILE_IDX,
                OFFSET
            ))
            break

    # print ("file {}, mean diff: {}".format(
    #     FILE_IDX,
    #     np.around(,3)
    # ))


# WHAT THE SERIOUS SHIT???


# file 0, offset 0
# file 1, offset 0
# file 2, offset 0
# file 3, offset 0
# file 4, offset 0
# file 5, offset 0
# file 6, offset 0
# file 7, offset 0
# file 8, offset 0
# file 9, offset 0
# file 10, offset 0
# file 11, offset 0
# file 12, offset 0
# file 13, offset 0
# file 14, offset 0
# file 15, offset 0
# file 16, offset 0
# file 17, offset 9
# file 18, offset 9
# file 19, offset 9
# file 20, offset 9
# file 21, offset 9
# file 22, offset 8
# file 23, offset 8
# file 24, offset 7
# file 25, offset 7
# file 26, offset 7
# file 27, offset 7
# file 28, offset 7
# file 29, offset 7
# file 30, offset 7
# file 31, offset 7
# file 32, offset 6
# file 33, offset 5
# file 34, offset 5
# file 35, offset 4
# file 36, offset 4
# file 37, offset 4
# file 38, offset 4
# file 39, offset 4
# file 40, offset 4
# file 41, offset 4
# file 42, offset 4
# file 43, offset 4
# file 44, offset 4
# file 45, offset 4
# file 46, offset 4
# file 47, offset 4
# file 48, offset 4
# file 49, offset 4
# file 50, offset 4
# file 51, offset 4
# file 52, offset 4
# file 53, offset 4
# file 54, offset 4
# file 55, offset 4
# file 56, offset 4
# file 57, offset 4
# file 58, offset 4
# file 59, offset 4
# file 60, offset 4
# file 61, offset 4
# file 62, offset 4
# file 63, offset 4
# file 64, offset 4
# file 65, offset 4
# file 66, offset 4
# file 67, offset 4
# file 68, offset 4
# file 69, offset 4
# file 70, offset 4
# file 71, offset 4
# file 72, offset 4
# file 73, offset 4
# file 74, offset 4
# file 75, offset 4
# file 76, offset 4
# file 77, offset 4
# file 78, offset 4
# file 79, offset 4
# file 80, offset 4
# file 81, offset 4
# file 82, offset 4
# file 83, offset 4
# file 84, offset 4
# file 85, offset 4
# file 86, offset 4
# file 87, offset 4
# file 88, offset 4
# file 89, offset 4
# file 90, offset 4
# file 91, offset 4
# file 92, offset 4
# file 93, offset 4
# file 94, offset 4
# file 95, offset 4
# file 96, offset 4
# file 97, offset 4
# file 98, offset 4
# file 99, offset 4
# file 100, offset 4
# file 101, offset 4
# file 102, offset 4
# file 103, offset 4
# file 104, offset 4
# file 105, offset 4
# file 106, offset 4
# file 107, offset 4
# file 108, offset 4
# file 109, offset 4
# file 110, offset 4
# file 111, offset 4
# file 112, offset 4
# file 113, offset 4
# file 114, offset 4
# file 115, offset 4
# file 116, offset 4
# file 117, offset 4
# file 118, offset 4
# file 119, offset 4
# file 120, offset 4
# file 121, offset 4
# file 122, offset 4
# file 123, offset 4
# file 124, offset 4
# file 125, offset 4
# file 126, offset 4
# file 127, offset 4
# file 128, offset 4
# file 129, offset 4
# file 130, offset 4
# file 131, offset 4
# file 132, offset 4
# file 133, offset 4
# file 134, offset 4
# file 135, offset 4
# file 136, offset 4
# file 137, offset 4
# file 138, offset 4
# file 139, offset 4
# file 140, offset 4
# file 141, offset 4
# file 142, offset 4
# file 143, offset 4
# file 144, offset 4
# file 145, offset 4
# file 146, offset 4
# file 147, offset 4
