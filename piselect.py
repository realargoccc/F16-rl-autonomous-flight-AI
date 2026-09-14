'''selector training env, decision pools includes offense (v4.1.6), defense(v1.0.2), h2h(v1.0.1)'''

import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from flight_env import F16Env
from selector import Selector 

DECISION_STEPS = 10 #1s per decision

class _Raw:
    mean, var = 0.0, 1.0

class SelectEnv(gym.Env):
    def __init__(self, aspect_band=(150.0, 180.0), range_band=(2500.0, 3500.0), defensive_p=0.3, bandit_p=0.5,
                 defense_override=True):
        self.env = F16Env()
        self.env.aspect_band = aspect_band
        self.env.range_band = range_band             
        self.env.defensive_p = defensive_p   
        self.env.foe_pool_prob = 1.0 - bandit_p     #share of fights against a selector opponent
        self.sel = Selector()                  
        self.defense_override = defense_override
        self.action_space = spaces.Discrete(len(Selector.MODES))
        self.observation_space = self.env.observation_space
        self.add_opponent(self.sel, "hand")     #hand turn 90 deg rule
