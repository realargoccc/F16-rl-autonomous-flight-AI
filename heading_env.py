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

        #no wings level, whatever the attitude left it
        self.me['ic/phi-deg'] = float(self.np_random.uniform(-45.0, 45.0))
        self.me['ic/theta-deg'] = float(self.np_random.uniform(-15.0, 15.0))
        self.me['ic/psi-true-deg'] = float(self.np_random.uniform(0.0, 360.0))
        self.me.run_ic()
        self.me.set_origin(self.me['position/lat-geod-deg'], self.me['position/long-gc-deg'])

        #300 steps at 0.1s
        self.curr_step = 0
        self.turn_counts = 0
        dt = self.me.get_delta_t() * self.sim_steps_per_action
        self.check_steps = int(round(self.check_interval / dt)) 

        self.tgt_hdg = self.me['attitude/psi-rad']
        self.tgt_alt = self.me['position/h-sl-meters']
        self.tgt_spd = self.me['velocities/u-fps'] * 0.3048
        self._new_target()

        return self._get_obs(), {}