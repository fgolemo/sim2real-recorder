import h5py

f = h5py.File("data/test-dataset.hdf5", "r")
episode_current_pos_real = f.get("episode_current_pos_real")
episode_current_vel_real = f.get("episode_current_vel_real")
episode_next_pos_sim = f.get("episode_next_pos_sim")
episode_next_vel_sim = f.get("episode_next_vel_sim")
episode_next_pos_real = f.get("episode_next_pos_real")
episode_next_vel_real = f.get("episode_next_vel_real")

episode_current_action = f.get("episode_current_action")
episode_current_speed = f.get("episode_current_speed")

episode_current_img_real = f.get("episode_current_img_real")
episode_next_img_real = f.get("episode_next_img_real")
episode_next_img_sim = f.get("episode_next_img_sim")

##### testing images

import matplotlib.pyplot as plt
import numpy as np

# plt.ion()
# img = np.zeros((256, 756, 3), dtype=np.uint8)
# plt_img = plt.imshow(img, interpolation='none', animated=True, label="blah")
# plt_ax = plt.gca()
#
# print (episode_current_img_real.shape)
# print(episode_next_img_real.shape)
# print(episode_next_img_sim.shape)
#
# for i in range(len(episode_current_img_real)):
#     img[:240, :250, :] = episode_current_img_real[i, :, :, :3]
#     img[:, 250:506, :] = episode_next_img_sim[i, :, :, :3]
#     img[:240, 506:, :] = episode_next_img_real[i, :, :, :3]
#
#     plt_img.set_data(img)
#     plt_ax.plot([0])
#     plt.pause(0.001)





###### testing positions

# for i in range(len(episode_current_pos_real)):
#     pos_real_t0 = episode_current_pos_real[i]
#     pos_real_t1 = episode_next_pos_real[i]
#     pos_sim_t1 = episode_next_pos_sim[i]
#     action = episode_current_action[i]
#
#     print ("{}\n{}\n{}\n{}\n\n".format(pos_real_t0, pos_real_t1, pos_sim_t1, action))

###### testing velocities

for i in range(len(episode_current_vel_real)):
    vel_real_t0 = episode_current_vel_real[i]
    vel_real_t1 = episode_next_vel_real[i]
    vel_sim_t1 = episode_next_vel_sim[i]
    vel_target = episode_current_speed[i]

    print ("{}\n{}\n{}\n{}\n\n".format(vel_real_t0, vel_real_t1, vel_sim_t1, vel_target))

