import gymnasium as gym
import numpy as np
import math
from gymnasium.spaces import Box, MultiDiscrete
from flight_env import Aircraft, decode_bins, THRO_BINS, SURF_BINS, THRO_LO, THRO_HI, SIM_STEPS_PER_ACTION

class HeadingEnv(gym.Env):
    '''fly to commanded heading (pretrained controller)'''

    def __init__(self):
        super().__init__()
        self.me = Aircraft()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
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
        self.alt_band = (4572.0, 10668.0)   #m, 15k - 35k ft
        self.ramp = [0.2, 0.4, 0.6, 0.8, 1.0]

        #reward scale
        self.hdg_scale = 20.0         #deg
        self.alt_scale = 400.0        #m, 50ft
        self.roll_scale = 45.0        #rad, 20 deg
        self.spd_scale = 50.0         #m/s

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

    def _new_target(self):
        '''a ramp that randomly increment whenever the run is succeed'''
        d = self.ramp[min(self.turn_counts, len(self.ramp) - 1)]
        self.tgt_hdg = (self.tgt_hdg + np.radians(self.np_random.uniform(-d, d) * self.max_hdg_step)) % (2 * np.pi)
        self.tgt_alt = float(np.clip(self.tgt_alt + self.np_random.uniform(-d, d) * self.max_alt_step, *self.alt_band))
        self.tgt_spd = float(np.clip(self.tgt_spd + self.np_random.uniform(-d, d) * self.max_spd_step, *self.spd_band))

    def _get_obs(self):
        alt        = self.me['position/h-sl-meters']
        roll       = self.me['attitude/phi-rad']
        pitch      = self.me['attitude/theta-rad']
        heading    = self.me['attitude/psi-rad']

        fwd_speed  = self.me['velocities/u-fps'] * 0.3048     #body x, nose
        side_speed = self.me['velocities/v-fps'] * 0.3048     #body y, right wing
        vert_speed = self.me['velocities/w-fps'] * 0.3048     #body z, belly
        ias        = self.me['velocities/vc-fps'] * 0.3048

        roll_rate  = self.me['velocities/p-rad_sec']
        pitch_rate = self.me['velocities/q-rad_sec']
        yaw_rate   = self.me['velocities/r-rad_sec']

        d_alt = self.tgt_alt - alt
        d_hdg = (self.tgt_hdg - heading + np.pi) % (2 * np.pi) - np.pi
        d_spd = self.tgt_spd - fwd_speed

        obs = np.array([
            d_alt / 3000.0, 
            d_hdg,
            d_spd / 100.0, 
            alt / 5000.0,
            np.sin(roll),
            np.cos(roll),
            np.sin(pitch),
            fwd_speed / 340.0,
            side_speed / 50.0,
            vert_speed / 50.0,
            ias / 340.0,
            roll_rate / np.pi,
            pitch_rate / 1.0,
            yaw_rate / 0.5,
            ], dtype=np.float32)
        return np.clip(obs, -10.0, 10.0)

    def _reward(self, crashed, deck_hit):
        '''geometric mean of four actions'''
        d_alt = abs(self.tgt_alt - self.me['position/h-sl-meters'])
        d_hdg = abs(math.degrees((self.tgt_hdg - self.me['attitude/psi-rad'] + np.pi) % (2 * np.pi) - np.pi))
        d_spd = abs(self.tgt_spd - self.me['velocities/u-fps'] * 0.3048)
        bank = abs(math.degrees(self.me['attitude/phi-rad']))

        r_hdg  = math.exp(-d_hdg / self.hdg_scale)
        r_alt  = math.exp(-d_alt / self.alt_scale)
        r_roll = math.exp(-bank  / self.roll_scale)
        r_spd  = math.exp(-d_spd / self.spd_scale)
        track  = (r_hdg * r_alt * r_roll * r_spd) ** 0.25

        r_term = -self.k_crash if (crashed or deck_hit) else 0.0
        self.last_terms = {"hdg": r_hdg, "alt": r_alt, "roll": r_roll,
                           "spd": r_spd, "track": track, "term": r_term}

        return track + r_term

    def decode(self, action):
        return decode_bins(action)
    
    def step(self, action):