import gymnasium as gym
import numpy as np
import math
from gymnasium.spaces import Box, MultiDiscrete
from flight_env import Aircraft, THRO_BINS, SURF_BINS, THRO_LO, THRO_HI, SIM_STEPS_PER_ACTION

class HeadingEnv(gym.Env):
    '''fly to commanded heading (pretrained controller)'''

    def __init__(self):
        super().__init__()
        self.me = Aircraft()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.action_space = MultiDiscrete([THRO_BINS, SURF_BINS, SURF_BINS, SURF_BINS])
        self.thro_bins, self.surf_bins = THRO_BINS, SURF_BINS
        self.thro_lo,   self.thro_hi   = THRO_LO,   THRO_HI

        self.sim_steps_per_action = SIM_STEPS_PER_ACTION
        self.max_episodes_steps = 1800      #180s run
        self.curr_step = 0

        #target generation
        self.check_interval = 30.0          #seconds
        self.pass_band = 10.0               #deg
        self.max_hdg_step = 180.0           #deg
        self.max_alt_step = 2134.0          #m, 7000 ft
        self.max_spd_step = 100.0           #m/s
        self.ramp = [0.2, 0.4, 0.6, 0.8, 1.0]

        #reward scale
        self.hdg_scale = 5.0          #deg
        self.alt_scale = 15.24        #m, 50ft
        self.roll_scale = 0.35        #rad, 20 deg
        self.spd_scale = 24.0         #m/s

        self.hard_deck = 2500.0
        self.k_crash = 300.0
        self.last_terms = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.me.fdm.reset_to_initial_conditions(0)  #jsbsim's run_ic
        self.me['ic/h-sl-ft'] = self.np_random.integers(18000, 25000)
        self.me['ic/vc-kts'] = 450.0
        self.me['ic/throttle-cmd-norm'] = 0.5
        self.me['ic/elevator-cmd-norm'] = 0.0
        self.me['gear/gear-cmd-norm'] = 0.0
        self.me['propulsion/tank[0]/contents-lbs'] = 1500.0
        self.me['propulsion/tank[1]/contents-lbs'] = 1500.0
        self.me['propulsion/engine/set-running'] = 1.0