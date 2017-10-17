SECONDS_OF_RECORDING = 3.0
NUMBER_OF_ACTIONS_PER_EPISODE = 3
MAX_ROBO_FPS = 100
ROBO_FPD_DELAY = 0.01
ROBO_REST_TIMER = 3.0
INITIAL_SPEED = 100
WRITE_EVERY_N_EPISODES = 5
SPEEDS = [50, 100, 200]
SPEED_STD_FACTOR = 4
TIME_MULTI = 1000000

USE_BACKUP = True
BACKUP_HOST = "tegra-ubuntu.local"
BACKUP_USER = "ubuntu"
BACKUP_PASS = "ubuntu"
BACKUP_PATH = "/mnt/2TB/flo-robot-data/"
BOUNDARIES = { # img is right-left flipped
    "top": 90,
    "left": 100,
    "right": 350,
    "bottom": 330
}
