"""
Settings for reward and training. Change values here, not in env.py, run.py or train.py.
"""


#showdown
SHOWDOWN_DIR = "SET-ME"  # where pokemon-showdown is cloned, use full path

# reward
VICTORY_VALUE = 1.0         # + win, - loss
FAINTED_VALUE = 0.3         # per pokemon fainted.
HP_VALUE = 0.1              # per full HP bar lost / dealt
STATUS_VALUE = 0.05         # per pokemon with status

# train
FORMAT = "gen9vgc2025regi"  # showdown format id
STEPS_PER_ENV = 512         # PPO n_step
TOTAL_TIMESTEPS = 84480     # steps per train.py run, not total.
SAVE_EVERY = 10240          # checkpoint to model/vgc.zip every this many steps

# run
RUN_NAME = "run3"               # model and logs saved under this name
TARGET_TIMESTEPS = 5_000_000    # training stops here
SEED = 0
