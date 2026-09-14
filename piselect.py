'''selector training env, decision pools includes offense (v4.1.6), defense(v1.0.2), h2h(v1.0.1)'''

import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from flight_env import F16Env
from selector import Selector, load_policy


DECISION_STEPS = 10 #1s per decision

class _Raw:
    mean, var = 0.0, 1.0

class LearnedSelector:
    def __init__(self, tag, sel, defense_override=True):
        self.chooser, self.rms, self.clip, self.eps = load_policy(tag)   #the chooser and its own obs normaliser
        self.sel = sel                                                  
        self.defense_override = defense_override
        self.reset()

    def reset(self):
        self.mode = None
        self.count = 0

    def predict(self, obs, deterministic=True):
        if self.count % DECISION_STEPS == 0:    #same decision logic as agent
            n = np.clip((obs - self.rms.mean) / np.sqrt(self.rms.var + self.eps), -self.clip, self.clip)
            choice, _ = self.chooser.predict(n.astype(np.float32), deterministic=deterministic)
            self.mode = Selector.MODES[int(choice)]
        self.count += 1
        defending = self.defense_override and self.sel.quadrant(obs) == "defense"
        return self.sel.act("defense" if defending else self.mode, obs, deterministic), None

class SelectEnv(gym.Env):
    def __init__(self, aspect_band=(150.0, 180.0), range_band=(2500.0, 3500.0), defensive_p=0.0, bandit_p=0.5,
                 defense_override=True):
        self.env = F16Env()
        self.env.load_foe("v4.1.6")
        self.env.aspect_band = aspect_band
        self.env.range_band = range_band             
        self.env.defensive_p = defensive_p   
        self.env.foe_pool_prob = 1.0 - bandit_p     #share of fights against a selector opponent
        self.sel = Selector()                  
        self.defense_override = defense_override
        self.action_space = spaces.Discrete(len(Selector.MODES))
        self.observation_space = self.env.observation_space

    def add_opponent(self, selector, tag):
        '''put selector in foe pool'''
        opp = copy.copy(selector)
        opp.reset()
        self.env.foe_pool.append((opp, _Raw, np.inf, 0.0))
        self.env.foe_tags.append(tag)
        self.env.foe_wins.append(0.0)
        self.env.foe_games.append(0.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.obs, info = self.env.reset(seed=seed)
        opp = self.env.foe_policy
        if opp is not None and hasattr(opp[0], "reset"):
            opp[0].reset()
        return self.obs, info

    def step(self, action):
        mode = Selector.MODES[int(action)]
        total = 0.0
        for _ in range(DECISION_STEPS):
            defending = self.defense_override and self.sel.quadrant(self.obs) == "defense"
            flown = "defense" if defending else mode
            self.obs, r, terminated, truncated, info = self.env.step(self.sel.act(flown, self.obs))
            total += float(r)
            if terminated or truncated:
                break
        return self.obs, total, terminated, truncated, info

def make_select_env(**kwargs):
    return Monitor(SelectEnv(**kwargs), info_keywords=("crashed", "foe_crashed", "win", "deck_hit", "foe_down"))